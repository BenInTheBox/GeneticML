pub fn relu(input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    input
        .iter()
        .map(|batch| {
            batch
                .iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect()
        })
        .collect()
}

pub fn sigmoid(input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    fn sigmoid_activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    input
        .iter()
        .map(|batch| batch.iter().map(|&x| sigmoid_activation(x)).collect())
        .collect()
}

pub fn tanh(input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    fn tanh_activation(x: f64) -> f64 {
        x.tanh()
    }

    input
        .iter()
        .map(|batch| batch.iter().map(|&x| tanh_activation(x)).collect())
        .collect()
}
