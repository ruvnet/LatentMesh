//! Ignored-by-default integration test against the *real* `agentbbs mcp`
//! stdio binary.
//!
//! Not run by `cargo test -p latentmesh-agentbbs-bridge` (no live binary is
//! assumed to be present in CI or in a fresh checkout) — this is the "add
//! one ignored-by-default integration test... if the build is quick" case
//! ADR-020's task described. It genuinely was quick: a shallow clone of
//! `github.com/ruvnet/agentbbs` (commit read at ADR-020 authoring time)
//! built `cargo build -p agentbbs --bin agentbbs` in well under a minute
//! (workspace warm-cache aside), and the resulting binary's `mcp`
//! subcommand was hand-verified against this crate's [`McpStdioClient`]
//! expectations before this test was written:
//!
//! ```text
//! $ printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n' \
//!     | ./agentbbs mcp
//! {"jsonrpc":"2.0","id":1,"result":{"capabilities":{...},
//!   "protocolVersion":"2024-11-05","serverInfo":{"name":"agentbbs-mcp",...}}}
//! ```
//!
//! ## How to run this test
//!
//! 1. Clone and build the binary (outside this workspace — agentbbs is not
//!    a dependency of it, per ADR-020):
//!    ```sh
//!    git clone --depth=1 https://github.com/ruvnet/agentbbs /tmp/agentbbs
//!    cd /tmp/agentbbs && cargo build -p agentbbs --bin agentbbs
//!    ```
//! 2. Point this test at the built binary and run it explicitly:
//!    ```sh
//!    AGENTBBS_MCP_BIN=/tmp/agentbbs/target/debug/agentbbs \
//!      cargo test -p latentmesh-agentbbs-bridge --test live_agentbbs_mcp -- --ignored
//!    ```
//!
//! If `AGENTBBS_MCP_BIN` is unset, the test skips itself with a clear
//! message rather than failing the run — this keeps `cargo test
//! --workspace` hermetic while still giving anyone with the binary on hand
//! a one-command way to exercise the real roundtrip.

use std::io::Read;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use latentmesh_agentbbs_bridge::mcp::{extract_tool_text, McpStdioClient};

/// A `Read` adapter over a child's stdout, so `McpStdioClient` (generic over
/// any `Read + Write` pair) can drive the subprocess directly.
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

#[test]
#[ignore = "spawns the real agentbbs mcp subprocess; set AGENTBBS_MCP_BIN and pass --ignored"]
fn live_agentbbs_mcp_roundtrip() {
    let Ok(bin) = std::env::var("AGENTBBS_MCP_BIN") else {
        eprintln!(
            "skipping: set AGENTBBS_MCP_BIN to a built `agentbbs` binary path \
             (see this file's module doc for build instructions)"
        );
        return;
    };

    let mut child = Command::new(&bin)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {bin} mcp: {e}"));

    let stdin: ChildStdin = child.stdin.take().expect("piped stdin");
    let stdout: ChildStdout = child.stdout.take().expect("piped stdout");
    let io = ChildIo { child, stdout };
    let mut client = McpStdioClient::new(io, stdin);

    let init = client.initialize().unwrap();
    assert_eq!(init["protocolVersion"], "2024-11-05");
    assert_eq!(init["serverInfo"]["name"], "agentbbs-mcp");

    let tools = client.list_tools().unwrap();
    let names: Vec<&str> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for expected in ["list_boards", "read_board", "post_message", "search_memory"] {
        assert!(names.contains(&expected), "missing tool: {expected}");
    }

    let boards = client.list_boards().unwrap();
    let text = extract_tool_text(&boards).expect("tool result has content[0].text");
    // No assertion on board contents (a fresh in-memory Bbs may start
    // empty); the roundtrip itself — spawn, initialize, call a tool, get a
    // well-shaped MCP result — is what this test proves.
    let _ = text;
}
