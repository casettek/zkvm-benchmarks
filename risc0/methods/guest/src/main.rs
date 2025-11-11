use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};
use half::bf16;

#[derive(Serialize, Deserialize)]
struct DotProductInput {
    column: Vec<u16>,
    vector: Vec<u16>,
}

fn main() {
    // Read the input containing two vectors (serialized as u16)
    let input: DotProductInput = env::read();

    // Validate that both vectors have the same length
    assert_eq!(
        input.column.len(),
        input.vector.len(),
        "Column and vector must have the same length"
    );

    // Convert u16 to bf16 and compute dot product
    // Accumulate in f32 to maintain precision during summation
    let dot_product: f32 = input
        .column
        .iter()
        .zip(input.vector.iter())
        .map(|(a, b)| {
            let a_bf16 = bf16::from_bits(*a);
            let b_bf16 = bf16::from_bits(*b);
            f32::from(a_bf16) * f32::from(b_bf16)
        })
        .sum();

    // Write the result to the journal
    env::commit(&dot_product);
}
