/// Castagnoli CRC-32C, reflected form, with the conventional initial and final
/// XOR. This table-free implementation is small enough for microcontrollers.
pub fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }
}
