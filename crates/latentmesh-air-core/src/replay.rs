use crate::{AirError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDecision {
    Accept,
    Duplicate,
    TooOld,
}

/// 64-message anti-replay window with well-defined `u16` wraparound.
/// Sequence jumps of exactly half the number space are rejected as ambiguous.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayWindow {
    initialized: bool,
    highest: u16,
    bitmap: u64,
}

impl ReplayWindow {
    pub const fn new() -> Self {
        Self {
            initialized: false,
            highest: 0,
            bitmap: 0,
        }
    }

    pub fn classify(&self, sequence: u16) -> ReplayDecision {
        if !self.initialized {
            return ReplayDecision::Accept;
        }
        let forward = sequence.wrapping_sub(self.highest);
        if forward != 0 && forward < 0x8000 {
            return ReplayDecision::Accept;
        }
        if forward == 0 {
            return ReplayDecision::Duplicate;
        }
        let behind = self.highest.wrapping_sub(sequence);
        if behind >= 64 {
            ReplayDecision::TooOld
        } else if self.bitmap & (1_u64 << behind) != 0 {
            ReplayDecision::Duplicate
        } else {
            ReplayDecision::Accept
        }
    }

    /// Commit only after complete reassembly and CRC validation. In-flight
    /// fragments sharing a message sequence must not be committed separately.
    pub fn commit(&mut self, sequence: u16) -> Result<()> {
        match self.classify(sequence) {
            ReplayDecision::Duplicate => return Err(AirError::Replay),
            ReplayDecision::TooOld => return Err(AirError::TooOld),
            ReplayDecision::Accept => {}
        }
        if !self.initialized {
            self.initialized = true;
            self.highest = sequence;
            self.bitmap = 1;
            return Ok(());
        }
        let forward = sequence.wrapping_sub(self.highest);
        if forward != 0 && forward < 0x8000 {
            self.bitmap = if forward >= 64 {
                1
            } else {
                (self.bitmap << forward) | 1
            };
            self.highest = sequence;
        } else {
            let behind = self.highest.wrapping_sub(sequence);
            self.bitmap |= 1_u64 << behind;
        }
        Ok(())
    }

    pub const fn highest(&self) -> Option<u16> {
        if self.initialized {
            Some(self.highest)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_reordering_once_and_rejects_replay() {
        let mut window = ReplayWindow::new();
        window.commit(10).unwrap();
        window.commit(12).unwrap();
        assert_eq!(window.classify(11), ReplayDecision::Accept);
        window.commit(11).unwrap();
        assert_eq!(window.classify(11), ReplayDecision::Duplicate);
        assert_eq!(window.commit(11), Err(AirError::Replay));
    }

    #[test]
    fn handles_u16_wraparound() {
        let mut window = ReplayWindow::new();
        window.commit(u16::MAX).unwrap();
        assert_eq!(window.classify(0), ReplayDecision::Accept);
        window.commit(0).unwrap();
        assert_eq!(window.highest(), Some(0));
        assert_eq!(window.classify(u16::MAX), ReplayDecision::Duplicate);
    }

    #[test]
    fn rejects_outside_window() {
        let mut window = ReplayWindow::new();
        window.commit(100).unwrap();
        assert_eq!(window.classify(36), ReplayDecision::TooOld);
    }
}
