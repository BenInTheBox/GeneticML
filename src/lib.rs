pub mod genetic_training;
pub mod neuralnetwork;

pub use genetic_training::*;
pub use neuralnetwork::*;

#[cfg(test)]
mod tests {
    use crate::layer::LinearLayer;

    #[test]
    fn test_linear_layer() {
        let mut layer = LinearLayer::new(3, 2);
        layer.set_weights(vec![vec![1., 2., 3.], vec![2., 3., 4.]], vec![0., 0.]);

        let input = vec![vec![1., 1., 1.], vec![0.5, 1., 1.5]];

        let output = layer.forward(&input);
        let target_output = vec![vec![6., 9.], vec![7., 10.]];

        assert_eq!(output, target_output);
    }
}
