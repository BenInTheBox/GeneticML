use rand::Rng;

const MAX_WEIGHT: f64 = 2.;

fn mutate_2d(weights: &mut Vec<Vec<f64>>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for weight in weights.iter_mut() {
        for w in weight.iter_mut() {
            if rng.gen::<f64>() < 0.5 {
                let pos_mult = if *w > 0. {
                    (1. - (*w / MAX_WEIGHT)).max(0.)
                } else {
                    1.
                };
                let neg_mult = if *w < 0. {
                    (1. + (*w / MAX_WEIGHT)).max(0.)
                } else {
                    1.
                };

                *w += rng
                    .gen_range(-(0.1 * neg_mult * mutation_rate)..(0.1 * pos_mult * mutation_rate));
            }
        }
    }
}

fn mutate_1d(weights: &mut Vec<f64>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for b in weights.iter_mut() {
        if rng.gen::<f64>() < 0.5 {
            let pos_mult = if *b > 0. {
                (1. - (*b / MAX_WEIGHT)).max(0.)
            } else {
                1.
            };
            let neg_mult = if *b < 0. {
                (1. + (*b / MAX_WEIGHT)).max(0.)
            } else {
                1.
            };

            *b += rng
                .gen_range(-(0.05 * neg_mult * mutation_rate)..(0.05 * pos_mult * mutation_rate));
        }
    }
}

#[derive(Clone)]
pub struct LinearLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let bias = vec![0.0; output_size];

        LinearLayer { weights, bias }
    }

    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>, bias: Vec<f64>) {
        self.weights = weights;
        self.bias = bias;
    }

    pub fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        input
            .iter()
            .map(|batch| {
                self.weights
                    .iter()
                    .enumerate()
                    .map(|(neuron_idx, weight)| {
                        let weighted_sum: f64 =
                            batch.iter().zip(weight.iter()).map(|(&x, &w)| x * w).sum();
                        weighted_sum + self.bias[neuron_idx]
                    })
                    .collect()
            })
            .collect()
    }

    pub fn mutate(&self, mutation_rate: f64) -> Self {
        let mut new_layer = self.clone();
        mutate_2d(&mut new_layer.weights, mutation_rate);
        mutate_1d(&mut new_layer.bias, mutation_rate);

        new_layer
    }
}
