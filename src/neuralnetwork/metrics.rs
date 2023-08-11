pub fn calculate_mse(target: &Vec<Vec<f64>>, prediction: &Vec<Vec<f64>>) -> f64 {
    let squared_errors: Vec<f64> = target
        .iter()
        .zip(prediction)
        .flat_map(|(target_batch, prediction_batch)| {
            target_batch
                .iter()
                .zip(prediction_batch)
                .map(|(&t, &p)| (t - p).powi(2))
        })
        .collect();

    let mean_squared_error = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;

    mean_squared_error
}