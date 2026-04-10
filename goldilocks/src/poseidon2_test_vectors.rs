extern crate alloc;
extern crate std;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_poseidon2::{MDSMat4, add_rc_and_sbox_generic, matmul_internal, mds_light_permutation};
use p3_symmetric::Permutation;

use crate::{
    GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS, GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8,
    GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
    GOLDILOCKS_POSEIDON2_RC_8_INTERNAL, Goldilocks, MATRIX_DIAG_8_GOLDILOCKS,
    default_goldilocks_poseidon2_8,
};

type F = Goldilocks;

const TOTAL_POSEIDON2_ROUNDS_8: usize = 30;
const POSEIDON2_ROUND_VECTOR_CASES: usize = 100;
const POSEIDON2_PERMUTATION_VECTOR_CASES: usize = 200;
const POSEIDON2_ROUND_VECTOR_REFERENCE_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/poseidon2_round_vector.txt");
const POSEIDON2_ROUND_VECTOR_GENERATED_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/poseidon2_round_vector_generated.txt");
const POSEIDON2_PERMUTATION_VECTOR_GENERATED_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/src/poseidon2_permutation_vector_generated.txt");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Poseidon2TraceStepKind {
    InitialLinearLayer,
    InitialFullRound,
    PartialRound,
    TerminalFullRound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Poseidon2Width8TraceEntry {
    step: usize,
    round: usize,
    kind: Poseidon2TraceStepKind,
    input: [F; 8],
    output: [F; 8],
}

fn trace_goldilocks_poseidon2_8(input: [F; 8]) -> Vec<Poseidon2Width8TraceEntry> {
    let mut state = input;
    let mut trace = Vec::with_capacity(
        1 + GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS
            + GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8
            + GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS,
    );
    let mut step = 0;

    let initial_linear_input = state;
    mds_light_permutation(&mut state, &MDSMat4);
    trace.push(Poseidon2Width8TraceEntry {
        step,
        round: 0,
        kind: Poseidon2TraceStepKind::InitialLinearLayer,
        input: initial_linear_input,
        output: state,
    });
    step += 1;

    for (round, rc) in GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL.iter().enumerate() {
        let round_input = state;
        state
            .iter_mut()
            .zip(rc.iter())
            .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
        mds_light_permutation(&mut state, &MDSMat4);
        trace.push(Poseidon2Width8TraceEntry {
            step,
            round,
            kind: Poseidon2TraceStepKind::InitialFullRound,
            input: round_input,
            output: state,
        });
        step += 1;
    }

    for (round, &rc) in GOLDILOCKS_POSEIDON2_RC_8_INTERNAL.iter().enumerate() {
        let round_input = state;
        add_rc_and_sbox_generic(&mut state[0], rc);
        matmul_internal(&mut state, MATRIX_DIAG_8_GOLDILOCKS);
        trace.push(Poseidon2Width8TraceEntry {
            step,
            round,
            kind: Poseidon2TraceStepKind::PartialRound,
            input: round_input,
            output: state,
        });
        step += 1;
    }

    for (round, rc) in GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL.iter().enumerate() {
        let round_input = state;
        state
            .iter_mut()
            .zip(rc.iter())
            .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
        mds_light_permutation(&mut state, &MDSMat4);
        trace.push(Poseidon2Width8TraceEntry {
            step,
            round,
            kind: Poseidon2TraceStepKind::TerminalFullRound,
            input: round_input,
            output: state,
        });
        step += 1;
    }

    trace
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn external_matrix_fallback_goldilocks_8(state: &mut [F; 8]) {
    const EXTERNAL_MDS_8: [[u64; 8]; 8] = [
        [10, 14, 2, 6, 5, 7, 1, 3],
        [8, 12, 2, 2, 4, 6, 1, 1],
        [2, 6, 10, 14, 1, 3, 5, 7],
        [2, 2, 8, 12, 1, 1, 4, 6],
        [5, 7, 1, 3, 10, 14, 2, 6],
        [4, 6, 1, 1, 8, 12, 2, 2],
        [1, 3, 5, 7, 2, 6, 10, 14],
        [1, 1, 4, 6, 2, 2, 8, 12],
    ];
    let input = *state;

    for (row, output) in EXTERNAL_MDS_8.iter().zip(state.iter_mut()) {
        let mut acc = F::ZERO;
        for (&coeff, input_elem) in row.iter().zip(input.iter()) {
            acc += *input_elem * F::from_u64(coeff);
        }
        *output = acc;
    }
}

fn apply_round_vector_reference_goldilocks_poseidon2_8(
    round_idx: usize,
    input: [F; 8],
) -> Option<[F; 8]> {
    let mut state = input;

    match round_idx {
        0..=3 => {
            let rc = &GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL[round_idx];
            state
                .iter_mut()
                .zip(rc.iter())
                .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
            external_matrix_fallback_goldilocks_8(&mut state);
            Some(state)
        }
        4..=25 => {
            let rc = GOLDILOCKS_POSEIDON2_RC_8_INTERNAL[round_idx - 4];
            add_rc_and_sbox_generic(&mut state[0], rc);
            matmul_internal(&mut state, MATRIX_DIAG_8_GOLDILOCKS);
            Some(state)
        }
        26..=29 => {
            let rc = &GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL[round_idx - 26];
            state
                .iter_mut()
                .zip(rc.iter())
                .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
            external_matrix_fallback_goldilocks_8(&mut state);
            Some(state)
        }
        _ => None,
    }
}

fn make_directed_round_vector(round_idx: usize, pattern: usize) -> [F; 8] {
    match pattern {
        0 => [F::ZERO; 8],
        1 => [F::ONE; 8],
        2 => [F::NEG_ONE; 8],
        3 => Goldilocks::new_array([
            0x0101010101010101 + round_idx as u64,
            0x0202020202020202 + round_idx as u64,
            0x0303030303030303 + round_idx as u64,
            0x0404040404040404 + round_idx as u64,
            0x0505050505050505 + round_idx as u64,
            0x0606060606060606 + round_idx as u64,
            0x0707070707070707 + round_idx as u64,
            0x0808080808080808 + round_idx as u64,
        ]),
        _ => {
            let round = (round_idx as u64) << 32;
            let pattern = (pattern as u64) << 8;
            Goldilocks::new_array([
                round | pattern,
                round | pattern | 1,
                round | pattern | 2,
                round | pattern | 3,
                round | pattern | 4,
                round | pattern | 5,
                round | pattern | 6,
                round | pattern | 7,
            ])
        }
    }
}

fn append_trace_state_line(line: &mut String, state: &[F; 8]) {
    for value in state {
        write!(line, " {:016x}", value.as_canonical_u64()).unwrap();
    }
}

fn make_directed_permutation_vector(pattern: usize) -> [F; 8] {
    match pattern {
        0 => [F::ZERO; 8],
        1 => [F::ONE; 8],
        2 => [F::NEG_ONE; 8],
        3 => Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7]),
        4 => Goldilocks::new_array([
            1u64 << 0,
            1u64 << 8,
            1u64 << 16,
            1u64 << 24,
            1u64 << 32,
            1u64 << 40,
            1u64 << 48,
            1u64 << 56,
        ]),
        5 => Goldilocks::new_array([
            0x5555555555555555,
            0xaaaaaaaaaaaaaaaa,
            0x5555555555555555,
            0xaaaaaaaaaaaaaaaa,
            0x5555555555555555,
            0xaaaaaaaaaaaaaaaa,
            0x5555555555555555,
            0xaaaaaaaaaaaaaaaa,
        ]),
        _ => unreachable!(),
    }
}

/// Full Poseidon2 permutation using the fallback 8×8 external MDS matrix
/// (matching the C reference implementation) instead of the optimised
/// `MDSMat4`-based light permutation.
fn permute_fallback_goldilocks_poseidon2_8(input: [F; 8]) -> [F; 8] {
    let mut state = input;

    // Initial linear layer
    external_matrix_fallback_goldilocks_8(&mut state);

    // Initial full rounds
    for rc in &GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL {
        state
            .iter_mut()
            .zip(rc.iter())
            .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
        external_matrix_fallback_goldilocks_8(&mut state);
    }

    // Partial rounds
    for &rc in &GOLDILOCKS_POSEIDON2_RC_8_INTERNAL {
        add_rc_and_sbox_generic(&mut state[0], rc);
        matmul_internal(&mut state, MATRIX_DIAG_8_GOLDILOCKS);
    }

    // Terminal full rounds
    for rc in &GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL {
        state
            .iter_mut()
            .zip(rc.iter())
            .for_each(|(s, &c)| add_rc_and_sbox_generic(s, c));
        external_matrix_fallback_goldilocks_8(&mut state);
    }

    state
}

fn generate_poseidon2_permutation_vector_text() -> String {
    let mut output = String::new();
    let mut rng_state = 0xDEADBEEFCAFE1234u64;

    for pattern in 0..6 {
        let input = make_directed_permutation_vector(pattern);
        let output_state = permute_fallback_goldilocks_poseidon2_8(input);
        append_trace_state_line(&mut output, &input);
        append_trace_state_line(&mut output, &output_state);
        output.push('\n');
    }

    for _ in 6..POSEIDON2_PERMUTATION_VECTOR_CASES {
        let input = Goldilocks::new_array(core::array::from_fn(|_| xorshift64(&mut rng_state)));
        let output_state = permute_fallback_goldilocks_poseidon2_8(input);
        append_trace_state_line(&mut output, &input);
        append_trace_state_line(&mut output, &output_state);
        output.push('\n');
    }

    output
}

fn generate_poseidon2_round_vector_text() -> String {
    let mut output = String::new();

    for round_idx in 0..TOTAL_POSEIDON2_ROUNDS_8 {
        for pattern in 0..5 {
            let input = make_directed_round_vector(round_idx, pattern);
            let round_output =
                apply_round_vector_reference_goldilocks_poseidon2_8(round_idx, input).unwrap();
            write!(&mut output, "{:02x}", round_idx).unwrap();
            append_trace_state_line(&mut output, &input);
            append_trace_state_line(&mut output, &round_output);
            output.push('\n');
        }

        let mut rng_state = 0xDEADBEEF12345678u64 ^ round_idx as u64;
        for _ in 0..(POSEIDON2_ROUND_VECTOR_CASES - 5) {
            let input = Goldilocks::new_array(core::array::from_fn(|_| xorshift64(&mut rng_state)));
            let round_output =
                apply_round_vector_reference_goldilocks_poseidon2_8(round_idx, input).unwrap();
            write!(&mut output, "{:02x}", round_idx).unwrap();
            append_trace_state_line(&mut output, &input);
            append_trace_state_line(&mut output, &round_output);
            output.push('\n');
        }
    }

    output
}

#[test]
fn test_trace_goldilocks_poseidon2_width_8_matches_permutation() {
    let input: [F; 8] = Goldilocks::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let trace = trace_goldilocks_poseidon2_8(input);

    assert_eq!(
        trace.len(),
        1 + GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS
            + GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8
            + GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS
    );
    assert_eq!(trace[0].input, input);
    assert_eq!(trace[0].kind, Poseidon2TraceStepKind::InitialLinearLayer);

    let mut expected = input;
    default_goldilocks_poseidon2_8().permute_mut(&mut expected);

    assert_eq!(trace.last().unwrap().output, expected);
}

#[test]
fn test_generate_poseidon2_round_vector_has_expected_line_count() {
    let generated = generate_poseidon2_round_vector_text();
    assert_eq!(generated.lines().count(), 3000);
}

#[test]
fn test_generate_poseidon2_permutation_vector_has_expected_line_count() {
    let generated = generate_poseidon2_permutation_vector_text();
    assert_eq!(generated.lines().count(), POSEIDON2_PERMUTATION_VECTOR_CASES);
}

#[test]
fn test_poseidon2_round_vector_matches_file() {
    let generated = generate_poseidon2_round_vector_text();
    let expected = std::fs::read_to_string(POSEIDON2_ROUND_VECTOR_REFERENCE_PATH).unwrap();
    if generated != expected {
        let generated_lines: Vec<_> = generated.lines().collect();
        let expected_lines: Vec<_> = expected.lines().collect();
        let mismatch = generated_lines
            .iter()
            .zip(expected_lines.iter())
            .position(|(lhs, rhs)| lhs != rhs);

        match mismatch {
            Some(line_idx) => panic!(
                "poseidon2 round vector mismatch at line {}:\nexpected: {}\ngenerated: {}",
                line_idx + 1,
                expected_lines[line_idx],
                generated_lines[line_idx],
            ),
            None => panic!(
                "poseidon2 round vector line count mismatch: expected {}, generated {}",
                expected_lines.len(),
                generated_lines.len(),
            ),
        }
    }
}

#[test]
#[ignore = "Run explicitly to refresh poseidon2_round_vector_generated.txt"]
fn write_generated_poseidon2_round_vector_file() {
    std::fs::write(
        POSEIDON2_ROUND_VECTOR_GENERATED_PATH,
        generate_poseidon2_round_vector_text(),
    )
    .unwrap();
}

#[test]
#[ignore = "Run explicitly to refresh poseidon2_permutation_vector_generated.txt"]
fn write_generated_poseidon2_permutation_vector_file() {
    std::fs::write(
        POSEIDON2_PERMUTATION_VECTOR_GENERATED_PATH,
        generate_poseidon2_permutation_vector_text(),
    )
    .unwrap();
}
