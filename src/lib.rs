pub mod genetic_training;
pub mod neuralnetwork;

pub use genetic_training::*;
pub use neuralnetwork::*;

#[cfg(test)]
mod tests {
    #[test]
    fn test_linear_layer() {}
}
