use rand::Rng;

pub fn w_dot_x(weights: &Vec<Vec<f64>>, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    inputs
        .iter()
        .map(|batch| {
            weights
                .iter()
                .map(|weight| batch.iter().zip(weight.iter()).map(|(&x, &w)| x * w).sum())
                .collect()
        })
        .collect()
}

pub fn m_addition(m1: &Vec<Vec<f64>>, m2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    m1.iter()
        .zip(m2.iter())
        .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(x1, x2)| x1 + x2).collect())
        .collect()
}

pub fn m_substraction(m1: &Vec<Vec<f64>>, m2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    m1.iter()
        .zip(m2.iter())
        .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(x1, x2)| x1 - x2).collect())
        .collect()
}

pub fn m_element_mul(m1: &Vec<Vec<f64>>, m2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    m1.iter()
        .zip(m2.iter())
        .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(x1, x2)| x1 * x2).collect())
        .collect()
}

pub fn add_bias(bias: &Vec<f64>, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    inputs
        .iter()
        .map(|batch| batch.iter().zip(bias.iter()).map(|(x, b)| x + b).collect())
        .collect()
}

const MAX_WEIGHT: f64 = 3.;

pub fn mutate_2d(weights: &mut Vec<Vec<f64>>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for weight in weights.iter_mut() {
        for w in weight.iter_mut() {
            if rng.gen::<f64>() < 0.1 {
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

pub fn mutate_1d(weights: &mut Vec<f64>, mutation_rate: f64) {
    let mut rng = rand::thread_rng();
    for b in weights.iter_mut() {
        if rng.gen::<f64>() < 0.2 {
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

pub fn w_random_init(input_size: usize, output_size: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let amplitude = (2.0 / (input_size + output_size) as f64).sqrt();

    (0..output_size)
        .map(|_| {
            (0..input_size)
                .map(|_| rng.gen_range(-amplitude..amplitude))
                .collect()
        })
        .collect()
}
