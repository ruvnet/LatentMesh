//! ADR-022 e2e loopback driver for `harness/integration`'s agentbbs bridge
//! suite: decode a `SemanticDelta`, run it through
//! `latentmesh-agentbbs-bridge`'s mapping functions (ADR-020), confirm the
//! resulting `post_message`/`ReplicateMessage` JSON matches the pinned
//! agentbbs contract shape; where the `agentbbs mcp` binary is present in
//! the environment (`AGENTBBS_MCP_BIN`, same convention as
//! `tests/live_agentbbs_mcp.rs`), additionally roundtrip it live over
//! stdio.
//!
//! Deterministic: a fixed Ed25519 seed ([`Identity::from_seed`], never
//! [`Identity::generate`]) and a fixed `created_at` timestamp so the emitted
//! JSON — and the receipt the Node harness runner builds from it — is
//! byte-reproducible across runs. Prints one JSON object to stdout; this
//! binary does not itself claim an evidence label.

use std::io::Read;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use chrono::{DateTime, Utc};
use latentmesh_agentbbs_bridge::bridge::{
    decode_delta, delta_to_post_args, delta_to_replicate_payload, sign_delta_message,
};
use latentmesh_agentbbs_bridge::mcp::{extract_tool_text, McpStdioClient};
use latentmesh_agentbbs_bridge::wire::{FederationPayload, Identity};
use latentmesh_air_core::{CriticalState, SemanticEnvelope, SymbolValue};

/// Fixed seed and timestamp, matching this crate's own hermetic golden-shape
/// tests (`tests/golden_payloads.rs`), so this driver's output is
/// deterministic across runs and across machines.
const IDENTITY_SEED: [u8; 32] = [9_u8; 32];

fn fixed_created_at() -> DateTime<Utc> {
    "2026-08-27T12:00:00Z".parse().unwrap()
}

fn sample_delta() -> latentmesh_air_core::SemanticDelta {
    let mut before = CriticalState::new();
    before.set(1, SymbolValue::Bool(true)).unwrap();
    before.set(2, SymbolValue::U64(827)).unwrap();
    let mut after = before.clone();
    after.set(2, SymbolValue::U64(828)).unwrap();
    after.set(3, SymbolValue::Q16_16(2 << 16)).unwrap();
    latentmesh_air_core::SemanticDelta::between(7, 3, 99, &before, &after, Vec::new()).unwrap()
}

/// A `Read` adapter over a child's stdout, mirroring
/// `tests/live_agentbbs_mcp.rs`'s `ChildIo`.
struct ChildIo {
    child: Child,
    stdout: ChildStdout,
}

impl Read for ChildIo {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stdout.read(buf)
    }
}

impl Drop for ChildIo {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Attempts the live MCP roundtrip named in ADR-022 when `AGENTBBS_MCP_BIN`
/// points at a built `agentbbs` binary. Returns `None` (not an error) when
/// the env var is unset — matching `tests/live_agentbbs_mcp.rs`'s
/// skip-not-fail convention, since no live binary is assumed present in a
/// fresh checkout or in CI.
fn try_live_mcp_roundtrip(board: &str, subject: &str, text: &str) -> Option<Result<(), String>> {
    let bin = std::env::var("AGENTBBS_MCP_BIN").ok()?;
    let mut child = Command::new(&bin)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;
    let stdin: ChildStdin = child.stdin.take().expect("piped stdin");
    let stdout: ChildStdout = child.stdout.take().expect("piped stdout");
    let io = ChildIo { child, stdout };
    let mut client = McpStdioClient::new(io, stdin);

    let result = (|| -> Result<(), String> {
        let init = client.initialize().map_err(|e| e.to_string())?;
        if init["protocolVersion"] != "2024-11-05" {
            return Err(format!("unexpected protocolVersion: {init}"));
        }
        let tools = client.list_tools().map_err(|e| e.to_string())?;
        let names: Vec<&str> = tools["tools"]
            .as_array()
            .ok_or("tools/list result missing tools array")?
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for expected in ["list_boards", "read_board", "post_message", "search_memory"] {
            if !names.contains(&expected) {
                return Err(format!("missing tool: {expected}"));
            }
        }
        let posted = client
            .post_message(board, subject, text)
            .map_err(|e| e.to_string())?;
        extract_tool_text(&posted).ok_or("post_message result missing content[0].text")?;
        Ok(())
    })();
    Some(result)
}

fn main() {
    let identity = Identity::from_seed(&IDENTITY_SEED);
    let created_at = fixed_created_at();
    let board = "air-relay";
    let handle = "gateway-1";

    let delta = sample_delta();
    let envelope = SemanticEnvelope::wrap_delta(&delta, 15, 0, None).unwrap();

    // Decode boundary: envelope -> delta, matching ADR-020's decode/re-encode
    // boundary (the envelope is already CRC-checked/authenticated upstream
    // by latentmesh-air-core before it would reach a real bridge).
    let decoded = decode_delta(&envelope).unwrap();
    let decode_round_trip_ok = decoded == delta;

    // Publish path 1: post_message MCP tool arguments (the "simplest,
    // human-board-facing" path).
    let post_args = delta_to_post_args(board, &delta);
    let post_args_shape_ok = post_args.get("board").and_then(|v| v.as_str()) == Some(board)
        && post_args
            .get("subject")
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("air-delta source="))
            .unwrap_or(false)
        && post_args
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("LatentMesh Air state delta"))
            .unwrap_or(false);

    // Publish path 2: signed message + federation-native ReplicateMessage.
    let message = sign_delta_message(&identity, board, &delta, handle, created_at).unwrap();
    let signature_verified = message.verify().is_ok();

    let replicate_payload =
        delta_to_replicate_payload(&identity, board, &delta, handle, created_at).unwrap();
    let replicate_json = serde_json::to_value(&replicate_payload).unwrap();
    let replicate_shape_ok = replicate_json.get("type").and_then(|v| v.as_str())
        == Some("replicate_message")
        && replicate_json.get("id").is_some()
        && replicate_json.get("signature").is_some()
        && replicate_json
            .get("body")
            .and_then(|b| b.get("board"))
            .and_then(|v| v.as_str())
            == Some(board);
    let replicate_verified = match &replicate_payload {
        FederationPayload::ReplicateMessage(m) => m.verify().is_ok(),
        _ => false,
    };

    let subject = post_args
        .get("subject")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let text = post_args
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let live_mcp = try_live_mcp_roundtrip(board, &subject, &text);
    let (mcp_attempted, mcp_ok, mcp_error) = match live_mcp {
        None => (false, false, None),
        Some(Ok(())) => (true, true, None),
        Some(Err(e)) => (true, false, Some(e)),
    };

    let output = serde_json::json!({
        "suite": "agentbbs-bridge-loopback-v1",
        "board": board,
        "decode_round_trip_ok": decode_round_trip_ok,
        "post_args_shape_ok": post_args_shape_ok,
        "post_args": post_args,
        "signature_verified": signature_verified,
        "replicate_shape_ok": replicate_shape_ok,
        "replicate_verified": replicate_verified,
        "mcp": {
            "attempted": mcp_attempted,
            "ok": mcp_ok,
            "error": mcp_error,
        },
    });

    println!("{output}");
}
