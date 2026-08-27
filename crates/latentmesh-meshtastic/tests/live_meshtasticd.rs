//! Ignored-by-default integration test against a *real* `meshtasticd`
//! device-API TCP server (ADR-019). Not run by `cargo test -p
//! latentmesh-meshtastic` (no live node is assumed present in CI or in a
//! fresh checkout) — same "add one ignored-by-default integration test
//! against the real thing, skip gracefully if the env var is unset"
//! pattern as `latentmesh-agentbbs-bridge/tests/live_agentbbs_mcp.rs`'s
//! `AGENTBBS_MCP_BIN`.
//!
//! This wraps the exact same handshake/round-trip flow as
//! `examples/meshtasticd_interop.rs` (both `#[path]`-include
//! `examples/meshtasticd_interop/core.rs` — see that file's module doc for
//! why it isn't just a shared library module) and asserts on the resulting
//! [`interop_core::InteropReport`] instead of printing it as JSON.
//!
//! ## How to run this
//!
//! ```sh
//! docker run -d --name meshtasticd-interop -p 4403:4403 \
//!   -v /path/to/config.yaml:/etc/meshtasticd/config.yaml:ro \
//!   meshtastic/meshtasticd:latest
//! MESHTASTICD_ADDR=127.0.0.1:4403 \
//!   cargo test -p latentmesh-meshtastic --test live_meshtasticd -- --ignored
//! ```
//!
//! If `MESHTASTICD_ADDR` is unset, the test skips itself with a clear
//! message rather than failing the run.

#[path = "../examples/meshtasticd_interop/core.rs"]
mod interop_core;

#[test]
#[ignore = "connects to a real meshtasticd TCP device API; set MESHTASTICD_ADDR and pass --ignored"]
fn live_meshtasticd_handshake_and_broadcast_round_trip() {
    let Ok(addr) = std::env::var("MESHTASTICD_ADDR") else {
        eprintln!(
            "skipping: set MESHTASTICD_ADDR (e.g. 127.0.0.1:4403) to a live meshtasticd \
             instance's device-API TCP port (see this file's module doc for the docker \
             one-liner)"
        );
        return;
    };

    let report = interop_core::run_interop(&addr);

    // --- Handshake: decoding real firmware-produced protobufs with our
    // hand-rolled codec is itself part of what this test validates.
    assert_ne!(
        report.handshake.my_node_num, 0,
        "expected a real MyNodeInfo.my_node_num from the handshake"
    );
    assert_eq!(
        report.handshake.config_complete_id,
        interop_core::WANT_CONFIG_ID,
        "config_complete_id must echo the id this test's want_config_id sent"
    );

    // --- Broadcast round-trips: these are the checks ADR-019's design
    // assumptions predicted would work, and they do, against real firmware.
    assert!(
        report.broadcast_single_fragment.byte_identical,
        "single-fragment broadcast round-trip did not reproduce the original bytes: {:?}",
        report.broadcast_single_fragment.reassembled_bytes
    );
    assert_eq!(report.broadcast_single_fragment.packet_count, 1);

    // --- Multi-fragment at the production MESHTASTIC_FRAME_MTU (227
    // bytes, revised down from ADR-019's original 233-byte assumption
    // after this exact live interop testing found 232+ bytes rejected with
    // Routing.Error.TOO_LARGE — see MESHTASTIC_FRAME_MTU's doc comment).
    // This is now required to round-trip byte-identical.
    assert!(
        report.broadcast_multi_fragment.byte_identical,
        "multi-fragment broadcast round-trip did not reproduce the original bytes"
    );
    assert_eq!(
        report.broadcast_multi_fragment.packet_count, 2,
        "300 bytes at the 211-usable-byte MTU must fragment into exactly 2 packets"
    );

    // --- MTU-ceiling regression tripwire: a raw 232-byte broadcast that
    // was confirmed rejected during the MESHTASTIC_FRAME_MTU boundary
    // search. Expected to stay rejected (`routed_back == false`); if this
    // ever flips, MESHTASTIC_FRAME_MTU may have room to move back up
    // towards 233 and the discrepancy text records that explicitly rather
    // than this test silently accepting the change.
    if report.mtu_ceiling_tripwire.routed_back {
        eprintln!(
            "NOTE: the {}-byte MTU-ceiling tripwire routed back this run — the previously \
             observed TOO_LARGE rejection may no longer hold; see report.discrepancies",
            report.mtu_ceiling_tripwire.payload_bytes
        );
        assert!(
            report
                .discrepancies
                .iter()
                .any(|d| d.contains("REGRESSION")),
            "tripwire routed back but no REGRESSION discrepancy was recorded"
        );
    }

    // --- Self-addressed unicast: ADR-019's task brief assumed this would
    // also route back through the API. Empirically (see this crate's
    // `data::SIMULATOR_APP_PORTNUM` doc comment and the discrepancy text
    // `run_interop` records) it does not, against this firmware/transport.
    // This is asserted as a known, recorded discrepancy rather than a
    // silent pass/fail — if a future meshtasticd version starts routing it
    // back, that's worth noticing too.
    if report.self_addressed_unicast.routed_back {
        eprintln!(
            "NOTE: self-addressed unicast routed back this run — the previously observed \
             discrepancy may no longer hold for this meshtasticd build; see report.discrepancies"
        );
    } else {
        assert!(
            !report.discrepancies.is_empty(),
            "self-addressed unicast did not route back, but no discrepancy was recorded"
        );
    }
}
