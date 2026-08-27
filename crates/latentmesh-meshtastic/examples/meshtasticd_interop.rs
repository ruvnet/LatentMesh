//! Opt-in live interop check against a real `meshtasticd` instance
//! (ADR-019). Unlike `examples/e2e_loopback.rs` (pure loopback simulation,
//! no socket, always runs), this example needs a real Meshtastic node
//! reachable over TCP and is gated on the `MESHTASTICD_ADDR` environment
//! variable — absent -> skip gracefully with an exit code of 0, matching
//! this repo's established live-endpoint convention
//! (`latentmesh-agentbbs-bridge/tests/live_agentbbs_mcp.rs`'s
//! `AGENTBBS_MCP_BIN`).
//!
//! ## How to run this
//!
//! Start a real Meshtastic firmware build (portduino, simulated `sim`
//! radio — no RF hardware needed) as a local device-API TCP server:
//! ```sh
//! docker run -d --name meshtasticd-interop -p 4403:4403 \
//!   -v /path/to/config.yaml:/etc/meshtasticd/config.yaml:ro \
//!   meshtastic/meshtasticd:latest
//! ```
//! then:
//! ```sh
//! MESHTASTICD_ADDR=127.0.0.1:4403 \
//!   cargo run -p latentmesh-meshtastic --example meshtasticd_interop
//! ```
//!
//! Prints one JSON receipt to stdout, then exits non-zero if any
//! evidence-bearing check (handshake decode, broadcast round-trips) failed
//! — a routing discrepancy that was *expected and recorded* (see
//! `discrepancies` in the output) does not itself fail the run.

#[path = "meshtasticd_interop/core.rs"]
mod interop_core;

use interop_core::InteropReport;

fn main() {
    let Ok(addr) = std::env::var("MESHTASTICD_ADDR") else {
        eprintln!(
            "skipping: set MESHTASTICD_ADDR (e.g. 127.0.0.1:4403) to a live meshtasticd \
             instance's device-API TCP port (see this file's module doc for the docker \
             one-liner)"
        );
        return;
    };

    let report = interop_core::run_interop(&addr);
    println!("{}", to_json(&report));

    // `run_interop` already panics on a hard handshake failure (no
    // FromRadio.my_info / no matching config_complete_id within the
    // timeout), so reaching here means the handshake itself succeeded;
    // firmware_version is the one handshake field that's allowed to be
    // legitimately absent (some firmware builds may not send `metadata`),
    // so it's reported but doesn't gate the exit code.
    //
    // `mtu_ceiling_tripwire.routed_back == true` also does NOT gate the
    // exit code here: it would mean the firmware now accepts a size this
    // adapter previously confirmed rejected, which is worth surfacing (it
    // is recorded as a discrepancy above) but is not itself evidence this
    // *adapter* is broken — the opposite, if anything.
    let hard_failures = !report.broadcast_single_fragment.byte_identical
        || !report.broadcast_multi_fragment.byte_identical;
    if hard_failures {
        eprintln!(
            "one or more evidence-bearing checks failed (see JSON above); this is distinct \
             from a recorded discrepancy, which is expected/documented behavior"
        );
        std::process::exit(1);
    }
}

fn to_json(report: &InteropReport) -> String {
    use serde_json_lite::{array, boolean, number, object, opt_string, string, Value};

    fn round_trip_json(rt: &interop_core::RoundTripReport) -> Value {
        object([
            ("message_bytes", number(rt.message_bytes as f64)),
            ("packet_count", number(rt.packet_count as f64)),
            ("routed_back", boolean(rt.routed_back)),
            ("byte_identical", boolean(rt.byte_identical)),
            (
                "reassembled_bytes",
                match &rt.reassembled_bytes {
                    Some(bytes) => number(bytes.len() as f64),
                    None => Value::Null,
                },
            ),
        ])
    }

    object([
        ("suite", string("meshtasticd-interop-v1")),
        (
            "evidence_label",
            string(
                "real Meshtastic firmware (portduino, simulated radio) over TCP device API \
                 — no RF, no over-the-air claim",
            ),
        ),
        ("meshtasticd_addr", string(&report.addr)),
        (
            "handshake",
            object([
                (
                    "firmware_version",
                    opt_string(report.handshake.firmware_version.as_deref()),
                ),
                (
                    "my_node_num",
                    string(&format!("0x{:08x}", report.handshake.my_node_num)),
                ),
                (
                    "config_complete_id",
                    number(report.handshake.config_complete_id as f64),
                ),
                ("frames_seen", number(report.handshake.frames_seen as f64)),
            ]),
        ),
        (
            "self_addressed_unicast",
            round_trip_json(&report.self_addressed_unicast),
        ),
        (
            "broadcast_single_fragment",
            round_trip_json(&report.broadcast_single_fragment),
        ),
        (
            "broadcast_multi_fragment",
            round_trip_json(&report.broadcast_multi_fragment),
        ),
        (
            "mtu_ceiling_tripwire",
            object([
                (
                    "payload_bytes",
                    number(report.mtu_ceiling_tripwire.payload_bytes as f64),
                ),
                (
                    "routed_back",
                    boolean(report.mtu_ceiling_tripwire.routed_back),
                ),
            ]),
        ),
        (
            "discrepancies",
            array(report.discrepancies.iter().map(|d| string(d)).collect()),
        ),
    ])
    .to_string()
}

/// A tiny, dependency-free JSON writer — same rationale as
/// `examples/e2e_loopback.rs`'s copy: this crate deliberately carries no
/// `serde_json` dependency (`Cargo.toml`), and an example only needs to
/// emit one small, fixed-shape object to stdout. Extended with `Array` and
/// an explicit `Null` (for an absent firmware version / reassembly) beyond
/// what `e2e_loopback.rs` needed.
mod serde_json_lite {
    pub enum Value {
        String(String),
        Number(f64),
        Bool(bool),
        Null,
        Array(Vec<Value>),
        Object(Vec<(&'static str, Value)>),
    }

    impl std::fmt::Display for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Value::String(s) => {
                    write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                }
                Value::Number(n) => {
                    if n.fract() == 0.0 && n.abs() < 1e15 {
                        write!(f, "{}", *n as i64)
                    } else {
                        write!(f, "{n}")
                    }
                }
                Value::Bool(b) => write!(f, "{b}"),
                Value::Null => write!(f, "null"),
                Value::Array(items) => {
                    write!(f, "[")?;
                    for (index, item) in items.iter().enumerate() {
                        if index > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{item}")?;
                    }
                    write!(f, "]")
                }
                Value::Object(fields) => {
                    write!(f, "{{")?;
                    for (index, (key, value)) in fields.iter().enumerate() {
                        if index > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "\"{key}\":{value}")?;
                    }
                    write!(f, "}}")
                }
            }
        }
    }

    pub fn string(s: &str) -> Value {
        Value::String(s.to_string())
    }
    pub fn opt_string(s: Option<&str>) -> Value {
        match s {
            Some(s) => Value::String(s.to_string()),
            None => Value::Null,
        }
    }
    pub fn number(n: f64) -> Value {
        Value::Number(n)
    }
    pub fn boolean(b: bool) -> Value {
        Value::Bool(b)
    }
    pub fn array(items: Vec<Value>) -> Value {
        Value::Array(items)
    }
    pub fn object<const N: usize>(fields: [(&'static str, Value); N]) -> Value {
        Value::Object(fields.into_iter().collect())
    }
}
