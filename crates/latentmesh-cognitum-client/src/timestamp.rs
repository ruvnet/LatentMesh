//! UTC Unix-seconds → ISO 8601 formatting, with no external date/time
//! dependency (the workspace does not carry `chrono` or `time`, and adding
//! one just for `YYYY-MM-DDTHH:MM:SSZ` formatting would be disproportionate
//! for a signing primitive that only needs whole-second precision, per the
//! `X-Device-Timestamp` examples in `cognitum-one/api`'s
//! `docs/seed-integration.md` — e.g. `2026-04-30T15:18:00Z`).
//!
//! The Gregorian calendar conversion is Howard Hinnant's `civil_from_days`
//! algorithm (<http://howardhinnant.github.io/date_algorithms.html>),
//! proleptic-Gregorian and correct for the full `i64` range this crate
//! could plausibly see (any date from year -32767 to 32767 that also fits
//! `i64` seconds); the fleet clock is never going to be pre-1970 in
//! practice, but the algorithm is exact there too so there is no reason to
//! special-case it away.

/// Format `unix_secs` (seconds since the Unix epoch, UTC) as
/// `YYYY-MM-DDTHH:MM:SSZ`.
pub fn unix_to_iso8601_utc(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Days-since-epoch (1970-01-01 = 0) → proleptic-Gregorian (year, month,
/// day). See the module doc for the algorithm's provenance.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64; // day-of-era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day-of-year, [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test vectors independently computed via `date -u -d '<iso>' +%s`.
    #[test]
    fn epoch_zero_is_the_unix_epoch() {
        assert_eq!(unix_to_iso8601_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn matches_the_seed_integration_doc_example() {
        // docs/seed-integration.md's worked X-Device-Timestamp example.
        assert_eq!(unix_to_iso8601_utc(1_777_562_280), "2026-04-30T15:18:00Z");
    }

    #[test]
    fn leap_day_2024_is_handled() {
        assert_eq!(unix_to_iso8601_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn century_non_leap_year_boundary() {
        // 2000 is a leap year (divisible by 400); this is a classic
        // Gregorian-conversion regression check.
        assert_eq!(unix_to_iso8601_utc(951_868_800), "2000-03-01T00:00:00Z");
    }

    #[test]
    fn year_boundary_end_of_day() {
        assert_eq!(unix_to_iso8601_utc(946_684_799), "1999-12-31T23:59:59Z");
    }

    #[test]
    fn i32_boundary_still_correct_in_i64() {
        assert_eq!(unix_to_iso8601_utc(2_147_483_647), "2038-01-19T03:14:07Z");
    }

    #[test]
    fn pre_epoch_dates_round_trip_through_div_euclid() {
        // -1 is the last second of 1969-12-31; div_euclid/rem_euclid must
        // floor rather than truncate for this to come out right.
        assert_eq!(unix_to_iso8601_utc(-1), "1969-12-31T23:59:59Z");
    }
}
