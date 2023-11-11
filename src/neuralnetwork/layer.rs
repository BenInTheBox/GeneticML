use serde_derive::{Serialize, Deserialize};

use crate::neuralnetwork::activation::{sigmoid, tanh};
use crate::neuralnetwork::linalgebra::{
    add_bias, m_addition, m_element_mul, m_substraction, mutate_1d, mutate_2d, w_dot_x,
    w_random_init,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = w_random_init(input_size, output_size);
        let bias = vec![0.0; output_size];

        LinearLayer { weights, bias }
    }

    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>, bias: Vec<f64>) {
        self.weights = weights;
        self.bias = bias;
    }

    pub fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let temp = w_dot_x(&self.weights, input);
        add_bias(&self.bias, &temp)
    }

    pub fn mutate(&self, mutation_rate: f64) -> Self {
        let mut new_layer = self.clone();
        let coef = (2.0 / (self.weights.len() + self.weights[0].len()) as f64).sqrt();
        mutate_2d(&mut new_layer.weights, mutation_rate * coef);
        mutate_1d(&mut new_layer.bias, mutation_rate);

        new_layer
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GRULayer {
    pub w_reset: Vec<Vec<f64>>,
    pub u_reset: Vec<Vec<f64>>,
    pub b_reset: Vec<f64>,

    pub w_update: Vec<Vec<f64>>,
    pub u_update: Vec<Vec<f64>>,
    pub b_update: Vec<f64>,

    pub w_candidate: Vec<Vec<f64>>,
    pub u_candidate: Vec<Vec<f64>>,
    pub b_candidate: Vec<f64>,

    pub hidden_state: Vec<Vec<f64>>,
}

impl GRULayer {
    pub fn new(input_size: usize, output_size: usize, batch_size: usize) -> Self {
        let w_reset = w_random_init(input_size, output_size);
        let u_reset = w_random_init(output_size, output_size);
        let b_reset = vec![0.0; output_size];

        let w_update = w_random_init(input_size, output_size);
        let u_update = w_random_init(output_size, output_size);
        let b_update = vec![0.0; output_size];

        let w_candidate = w_random_init(input_size, output_size);
        let u_candidate = w_random_init(output_size, output_size);
        let b_candidate = vec![0.0; output_size];

        let hidden_state = vec![vec![0.0; output_size]; batch_size];

        GRULayer {
            w_reset,
            u_reset,
            b_reset,
            w_update,
            u_update,
            b_update,
            w_candidate,
            u_candidate,
            b_candidate,
            hidden_state,
        }
    }

    pub fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let z = sigmoid(add_bias(
            &self.b_update,
            &m_addition(
                &w_dot_x(&self.w_update, &input),
                &w_dot_x(&self.u_update, &self.hidden_state),
            ),
        ));

        let r = sigmoid(add_bias(
            &self.b_reset,
            &m_addition(
                &w_dot_x(&self.w_reset, &input),
                &w_dot_x(&self.u_reset, &self.hidden_state),
            ),
        ));

        let h_candidate = tanh(add_bias(
            &self.b_candidate,
            &m_addition(
                &w_dot_x(&self.w_candidate, &input),
                &w_dot_x(&self.u_candidate, &m_element_mul(&r, &self.hidden_state)),
            ),
        ));

        self.hidden_state = m_addition(
            &m_element_mul(
                &m_substraction(&vec![vec![1.; z[0].len()]; z.len()], &z),
                &self.hidden_state,
            ),
            &m_element_mul(&z, &h_candidate),
        );

        self.hidden_state.clone()
    }

    pub fn mutate(&self, mutation_rate: f64) -> Self {
        let mut new_layer = self.clone();
        let coef = (1.0 / (self.w_reset.len() + self.w_reset[0].len()) as f64).sqrt();
        mutate_2d(&mut new_layer.w_reset, mutation_rate * coef);
        mutate_2d(&mut new_layer.w_update, mutation_rate * coef);
        mutate_2d(&mut new_layer.w_candidate, mutation_rate * coef);

        mutate_1d(&mut new_layer.b_reset, mutation_rate);
        mutate_1d(&mut new_layer.b_update, mutation_rate);
        mutate_1d(&mut new_layer.b_candidate, mutation_rate);

        new_layer
    }

    pub fn reset(&mut self) {
        self.hidden_state = vec![vec![0.0; self.hidden_state[0].len()]; self.hidden_state.len()];
    }
}
