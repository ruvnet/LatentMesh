//! Integer piecewise-linear half-life decay and registered task weights.

pub const HALF_LIFE_MS: u64 = 1000;
pub const STALE_AGE_MS: u64 = 2000;
pub const USEFUL_COVERAGE_PPM: u64 = 900_000;
pub const WEIGHT_SYMBOL: u64 = 1;
pub const WEIGHT_RELEVANT_RESIDUAL: u64 = 4;

/// `D_ppm(a)`: halves every `HALF_LIFE_MS`, linear inside each half-life,
/// zero once `a > STALE_AGE_MS`.
pub fn decay_ppm(age_ms: u64) -> u64 {
    if age_ms > STALE_AGE_MS {
        return 0;
    }
    let base = 1_000_000_u64 >> (age_ms / HALF_LIFE_MS);
    base - (base / 2) * (age_ms % HALF_LIFE_MS) / HALF_LIFE_MS
}

/// Sum of weights of the registered relevant fields (all symbols plus the
/// relevant residual slots).
pub fn max_weight(symbols: u64, relevant_residuals: u64) -> u64 {
    symbols * WEIGHT_SYMBOL + relevant_residuals * WEIGHT_RELEVANT_RESIDUAL
}

/// `W_delivered >= 0.9 * W_max`, cross-multiplied.
pub fn covers_useful(w_delivered: u64, w_max: u64) -> bool {
    w_delivered * 1_000_000 >= USEFUL_COVERAGE_PPM * w_max
}
