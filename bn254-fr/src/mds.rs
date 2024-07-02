use std::sync::OnceLock;

use p3_field::AbstractField;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;
use p3_mds::{karatsuba_convolution::Convolve, util::first_row_to_first_col, MdsPermutation};
use crate::{Bn254Fr};

#[derive(Clone, Debug, Default)]
pub struct MdsMatrixBn254Fr;

// pub struct SmallConvolveBn254Fr;
// impl Convolve<Bn254Fr, i128, i64, i128> for SmallConvolveBn254Fr {
//     /// Return the lift of a Goldilocks element, 0 <= input.value <= P
//     /// < 2^64. We widen immediately, since some valid Goldilocks elements
//     /// don't fit in an i64, and since in any case overflow can occur
//     /// for even the smallest convolutions.
//     #[inline(always)]
//     fn read(input: Bn254Fr) -> [u64;4] {
//         input.value
//     }

//     /// For a convolution of size N, |x| < N * 2^64 and (as per the
//     /// assumption above), |y| < 2^51. So the product is at most N *
//     /// 2^115 which will not overflow for N <= 16. We widen `y` at
//     /// this point to perform the multiplication.
//     #[inline(always)]
//     fn parity_dot<const N: usize>(u: [i128; N], v: [i64; N]) -> i128 {
//         let mut s = 0i128;
//         for i in 0..N {
//             s += u[i] * v[i] as i128;
//         }
//         s
//     }

//     /// The assumptions above mean z < N^2 * 2^115, which is at most
//     /// 2^123 when N <= 16.
//     ///
//     /// NB: Even though intermediate values could be negative, the
//     /// output must be non-negative since the inputs were
//     /// non-negative.
//     #[inline(always)]
//     fn reduce(z: i128) -> Goldilocks {
//         debug_assert!(z >= 0);
//         reduce128(z as u128)
//     }
// }

// const MATRIX_CIRC_MDS_3_SML_ROW: [i64; 3] = [7, 1, 3];

// impl Permutation<[Bn254Fr; 3]> for MdsMatrixBn254Fr {
//     fn permute(&self, input: [Bn254Fr; 3]) -> [Bn254Fr; 3] {
//         const MATRIX_CIRC_MDS_3_SML_COL: [i64; 3] =
//             first_row_to_first_col(&MATRIX_CIRC_MDS_3_SML_ROW);
//         SmallConvolveBn254Fr::apply(
//             input,
//             MATRIX_CIRC_MDS_8_SML_COL,
//             SmallConvolveGoldilocks::conv8,
//         )
//     }

//     fn permute_mut(&self, input: &mut [Bn254Fr; 3]) {
//         *input = self.permute(*input);
//     }
// }
// impl MdsPermutation<Bn254Fr, 3> for MdsMatrixBn254Fr {}


// Use diffusion matrix as Mds matrix for testing poseidon over BN254Fr
#[inline]
fn get_diffusion_matrix_3() -> &'static [Bn254Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bn254Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()])
}

impl Permutation<[Bn254Fr; 3]> for MdsMatrixBn254Fr {
    fn permute_mut(&self, state: &mut [Bn254Fr; 3]) {
        matmul_internal::<Bn254Fr, Bn254Fr, 3>(state, *get_diffusion_matrix_3());
    }
}

impl MdsPermutation<Bn254Fr, 3> for MdsMatrixBn254Fr {}