extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_scratch;
use crate::genetic_rl::neuralnetwork::layer::LinearLayer;
use crate::genetic_rl::neuralnetwork::activation::{sigmoid, relu};
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
        NeuralNet{
            layer1: self.layer1.mutate(mutation_rate),
            layer2: self.layer2.mutate(mutation_rate),
        }
    }
}


#[derive(Clone)]
struct XorNot<A: Agent> {
    agent: A,
    pub inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
}

unsafe impl<A: Agent> Send for XorNot<A> {}
unsafe impl<A: Agent> Sync for XorNot<A> {}

impl<A: Agent> Simulation<A> for XorNot<A> {
    fn new(agent: A) -> Self {
        XorNot {
            agent,
            inputs: vec![
                vec![0., 0.],
                vec![1., 0.],
                vec![0., 1.],
                vec![1., 1.]
            ],
            targets: vec![
                vec![0.],
                vec![1.],
                vec![1.],
                vec![0.],
            ]
        }
    }

    fn evaluate_agent(&mut self) -> f64 {
        let prediction = self.agent.step(&self.inputs);

        - calculate_mse(&self.targets, &prediction)
    }

    fn get_agent(&self) -> A {
        self.agent.clone()
    }

    fn new_env(&self, agent: A) -> Self {
        XorNot::new(agent)
    }

    fn on_generation(&mut self, _generation_number: usize) {
    }
}

pub fn main() {
    let nb_generation: usize = 1000;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.05;

    let mutation_rate: f64 = 0.3;
    let mutation_decay: f64 = 0.999;

    let mut population = training_from_scratch::<NeuralNet, XorNot<NeuralNet>>(
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

    let sim = XorNot::new(population[0].clone());
    println!("{:?}", population[0].step(&sim.targets));
}