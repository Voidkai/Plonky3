use alloc::vec;
use alloc::vec::Vec;

use modinverse::modinverse;
use p3_field::PrimeField64;
use p3_util::relatively_prime_u64;
use sha3::Shake256;
use sha3::digest::{ExtendableOutput, Update, XofReader};

/// Generate alpha, the smallest integer relatively prime to `p − 1`.
pub(crate) const fn get_alpha<F: PrimeField64>() -> u64 {
    let p = F::ORDER_U64;
    let mut a = 3;

    while a < p {
        if relatively_prime_u64(a, p - 1) {
            return a;
        }
        a += 1;
    }

    panic!("No valid alpha found. Rescue does not support fields of order 2 or 3.");
}

/// Given alpha, find its multiplicative inverse in `Z/⟨p − 1⟩`.
pub(crate) fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64 as i128;
    modinverse(alpha as i128, p - 1)
        .expect("x^alpha not a permutation")
        .unsigned_abs()
        .try_into()
        .unwrap()
}

/// Compute the SHAKE256 variant of SHA-3.
/// This is used to generate the round constants from a seed string.
pub(crate) fn shake256_hash(seed_bytes: &[u8], num_bytes: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; num_bytes];
    reader.read(&mut result);
    result
}
