//! Injectable time source for canonical-request signing (ADR-021).
//!
//! [`crate::signing::DeviceIdentity::sign`] takes a `&dyn Clock` instead of
//! calling `SystemTime::now()` internally, so a golden signing test can pin
//! the timestamp and get a byte-exact, reproducible signature.

/// A source of the current time, expressed as Unix epoch seconds (UTC).
pub trait Clock {
    fn now_unix(&self) -> i64;
}

/// The real wall clock, backed by `SystemTime::now()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the Unix epoch")
            .as_secs() as i64
    }
}

/// A fixed point in time. Used by tests that need a byte-exact signature,
/// and available to any caller that wants to control exactly what
/// timestamp gets signed (e.g. to reuse a timestamp captured earlier in a
/// retry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedClock(pub i64);

impl Clock for FixedClock {
    fn now_unix(&self) -> i64 {
        self.0
    }
}

/// The server rejects a signed request when
/// `|server_now - X-Device-Timestamp| > 300s`. Source:
/// `cognitum-one/api`'s `openapi/cognitum-api.yaml`
/// (`components.securitySchemes.ed25519`) and `docs/seed-integration.md`
/// ("Authentication" section: "rejects requests where
/// `|server_now - X-Device-Timestamp| > 300s`"); transcribed into
/// ADR-021's Decision section. 300s is inclusive of the boundary itself —
/// only a skew strictly greater than this is rejected.
pub const MAX_CLOCK_SKEW_SECS: i64 = 300;

/// The server also rejects a duplicate signature seen again within a
/// 10-minute window (`docs/seed-integration.md`: "rejects duplicate
/// signatures within a 10-minute replay window (Firestore TTL'd)"). This is
/// a server-side dedup concern — there is no client-side algorithm to
/// implement, since the server is the one deduplicating signatures it has
/// already seen. The constant is recorded here as the contract a caller
/// must respect (never reuse a [`crate::signing::SignedRequest`] — always
/// sign fresh) and as a fixture for a boundary-value test.
pub const REPLAY_WINDOW_SECS: i64 = 600;

/// Would the server accept a request signed at `request_timestamp_unix` if
/// its own clock reads `server_now_unix`? Mirrors the server's `> 300`
/// rejection exactly (see [`MAX_CLOCK_SKEW_SECS`]).
pub fn is_within_clock_skew(server_now_unix: i64, request_timestamp_unix: i64) -> bool {
    (server_now_unix - request_timestamp_unix).abs() <= MAX_CLOCK_SKEW_SECS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_boundary_is_inclusive_at_300() {
        assert!(is_within_clock_skew(1_000_300, 1_000_000));
        assert!(is_within_clock_skew(1_000_000, 1_000_300));
    }

    #[test]
    fn skew_boundary_rejects_301() {
        assert!(!is_within_clock_skew(1_000_301, 1_000_000));
        assert!(!is_within_clock_skew(1_000_000, 1_000_301));
    }

    #[test]
    fn replay_window_constant_matches_documented_value() {
        assert_eq!(REPLAY_WINDOW_SECS, 600);
    }

    #[test]
    fn fixed_clock_returns_the_same_value_every_call() {
        let clock = FixedClock(1_777_562_280);
        assert_eq!(clock.now_unix(), 1_777_562_280);
        assert_eq!(clock.now_unix(), 1_777_562_280);
    }

    #[test]
    fn system_clock_returns_a_plausible_unix_timestamp() {
        // Sanity check only — not a golden test. 2020-01-01 as a floor
        // catches a badly broken clock without pinning an exact value.
        assert!(SystemClock.now_unix() > 1_577_836_800);
    }
}
