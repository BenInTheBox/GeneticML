extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_checkpoint;
use crate::genetic_rl::neuralnetwork::activation::{relu, sigmoid};
use crate::genetic_rl::neuralnetwork::layer::LinearLayer;
use crate::genetic_rl::neuralnetwork::metrics::calculate_mse;

#[derive(Clone)]
pub struct NeuralNet {
    layer1: LinearLayer,
    layer2: LinearLayer,
}

unsafe impl Send for NeuralNet {}
unsafe impl Sync for NeuralNet {}

impl Agent for NeuralNet {
    fn new() -> Self {
        NeuralNet {
            layer1: LinearLayer::new(2, 2),
            layer2: LinearLayer::new(2, 1),
        }
    }

    fn reset(&mut self) {}

    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = self.layer1.forward(input);
        output = relu(output);

        output = self.layer2.forward(&output);
        sigmoid(output)
    }

    fn mutate(&self, mutation_rate: f64) -> Self {
        NeuralNet {
            layer1: self.layer1.mutate(mutation_rate),
            layer2: self.layer2.mutate(mutation_rate),
        }
    }
}

#[derive(Clone)]
struct XorNot {
    pub inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
}

impl Simulation for XorNot {

    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent,
    {
        let prediction = agent.step(&self.inputs);

        -calculate_mse(&self.targets, &prediction)
    }

    fn on_generation(&mut self, _generation_number: usize) {}
}

impl XorNot {
    fn new() -> Self {
        XorNot {
            inputs: vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]],
            targets: vec![vec![0.], vec![1.], vec![1.], vec![0.]],
        }
    }
}

pub fn main() {
    let nb_generation: usize = 1000;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.05;

    let mutation_rate: f64 = 0.3;
    let mutation_decay: f64 = 0.999;

    let mut simulation = XorNot::new();
    let mut population: Vec<NeuralNet> = (0..nb_individus).map(|_| NeuralNet::new()).collect();

    population = training_from_checkpoint::<NeuralNet, XorNot>(
        population,
        &mut simulation,
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    println!("{:?}", population[0].layer1.weights);
    println!("{:?}", population[0].layer1.bias);
    println!("{:?}", population[0].layer2.weights);
    println!("{:?}", population[0].layer2.bias);

    println!("{:?}", population[0].step(&simulation.targets));
}
