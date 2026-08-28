//! Injected-vs-natural layer-k norm instrumentation (design §3, S0 norm-band
//! gate, risk #5). Pure CPU statistics over f32 vectors.

use serde::Serialize;

/// L2 norm of a vector.
pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Summary statistics of a norm sample (percentiles by nearest-rank).
#[derive(Debug, Clone, Serialize)]
pub struct NormStats {
    pub n: usize,
    pub min: f32,
    pub p25: f32,
    pub median: f32,
    pub p75: f32,
    pub max: f32,
    pub mean: f32,
}

pub fn stats(mut values: Vec<f32>) -> NormStats {
    assert!(!values.is_empty(), "norm stats over empty sample");
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let pct = |q: f64| values[(((n - 1) as f64) * q).round() as usize];
    NormStats {
        n,
        min: values[0],
        p25: pct(0.25),
        median: pct(0.5),
        p75: pct(0.75),
        max: values[n - 1],
        mean: values.iter().sum::<f32>() / n as f32,
    }
}

/// S0 norm-band gate: is `x` within `[median/factor, median*factor]` of the
/// natural distribution's median? (design: "within ~3x", factor = 3.0).
pub fn within_band(x: f32, natural_median: f32, factor: f32) -> bool {
    x >= natural_median / factor && x <= natural_median * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_and_band() {
        let s = stats(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.median, 3.0);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert!(within_band(1.0, 3.0, 3.0));
        assert!(within_band(9.0, 3.0, 3.0));
        assert!(!within_band(9.1, 3.0, 3.0));
        assert!(!within_band(0.9, 3.0, 3.0));
    }

    #[test]
    fn l2_norm() {
        assert!((l2(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }
}
