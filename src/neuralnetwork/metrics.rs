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

pub fn calculate_mse_time_series(
    target: &Vec<Vec<Vec<f64>>>,
    prediction: &Vec<Vec<Vec<f64>>>,
) -> f64 {
    let squared_errors: Vec<f64> =
        target
            .iter()
            .zip(prediction.iter())
            .flat_map(|(target_t, prediction_t)| {
                target_t.iter().zip(prediction_t.iter()).flat_map(
                    |(target_batch, prediction_batch)| {
                        target_batch.iter().zip(prediction_batch.iter()).map(
                            |(&target_val, &prediction_val)| (target_val - prediction_val).abs(),
                        )
                    },
                )
            })
            .collect();

    let mean_squared_error = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;

    mean_squared_error
}
