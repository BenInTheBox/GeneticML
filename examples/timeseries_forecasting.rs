extern crate genetic_rl;

use serde_derive::{Serialize, Deserialize};
use std::fs;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_checkpoint;
use crate::genetic_rl::neuralnetwork::activation::tanh;
use crate::genetic_rl::neuralnetwork::layer::{GRULayer, LinearLayer};
use crate::genetic_rl::neuralnetwork::metrics::calculate_mse_time_series;

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralNet {
    layer1: GRULayer,
    layer2: GRULayer,
    layer3: LinearLayer,
}

unsafe impl Send for NeuralNet {}
unsafe impl Sync for NeuralNet {}

impl Agent for NeuralNet {
    fn reset(&mut self) {
        self.layer1.reset();
        self.layer2.reset();
    }

    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = self.layer1.forward(input);

        output = self.layer2.forward(&output);
        output = self.layer3.forward(&output);

        tanh(output)
    }

    fn mutate(&self, mutation_rate: f64) -> Self {
        NeuralNet {
            layer1: self.layer1.mutate(mutation_rate),
            layer2: self.layer2.mutate(mutation_rate),
            layer3: self.layer3.mutate(mutation_rate),
        }
    }
}

impl NeuralNet {
    fn new() -> Self {
        NeuralNet {
            layer1: GRULayer::new(1, 7, 1),
            layer2: GRULayer::new(7, 5, 1),
            layer3: LinearLayer::new(5, 1),
        }
    }
}

#[derive(Clone)]
struct Timeserie {
    pub inputs: Vec<Vec<Vec<f64>>>,
    targets: Vec<Vec<Vec<f64>>>,
}

impl Simulation for Timeserie {
    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent,
    {
        agent.reset();
        let memory_loading_steps: usize = 50;

        self.inputs
            .iter()
            .take(memory_loading_steps)
            .for_each(|input_i| {
                let _ = agent.step(input_i);
            });

        let prediction: Vec<Vec<Vec<f64>>> = self
            .inputs
            .iter()
            .skip(memory_loading_steps)
            .map(|input_i| agent.step(input_i))
            .collect();

        -calculate_mse_time_series(
            &self
                .targets
                .iter()
                .skip(memory_loading_steps)
                .map(|x| x.iter().map(|y| y.iter().map(|z| *z).collect()).collect())
                .collect(),
            &prediction,
        )
    }

    fn on_generation(&mut self, _generation_number: usize) {}
}

impl Timeserie {
    fn new() -> Self {
        let dt: f64 = 1. / 6.;
        let amplitude: f64 = 0.3;
        let total_steps: usize = (100. / dt) as usize;

        let timeserie: Vec<f64> = (0..total_steps)
            .map(|step| {
                amplitude * ((step as f64 * dt).sin() + (step as f64 * dt * 2.).sin())
                    + 0.05 * (step as f64 * dt * 8.1).sin()
                    + 0.02 * (step as f64 * dt * 16.1).sin()
            })
            .collect();

        Timeserie {
            inputs: timeserie
                .iter()
                .take(timeserie.len() - 1)
                .map(|x| vec![vec![*x]])
                .collect(),
            targets: timeserie.iter().skip(1).map(|x| vec![vec![*x]]).collect(),
        }
    }

    pub fn forecast<A>(
        &self,
        agent: &mut A,
        horizon: usize,
        current_input: &Vec<Vec<f64>>,
    ) -> Vec<Vec<Vec<f64>>>
    where
        A: Agent,
    {
        let mut forecast: Vec<Vec<Vec<f64>>> = vec![];
        let mut last_prediction: Vec<Vec<f64>> = current_input.clone();

        (0..horizon).for_each(|_| {
            last_prediction = agent.step(&last_prediction);
            forecast.push(last_prediction.clone());
        });

        forecast
    }

    pub fn test_forecast<A>(&mut self, agent: &mut A, horizon: usize, forecast_start: usize)
    where
        A: Agent,
    {
        agent.reset();

        self.inputs.iter().take(forecast_start).for_each(|input_i| {
            let _ = agent.step(input_i);
        });

        let forecast = self.forecast(agent, horizon, &self.inputs[forecast_start + 1]);

        let targets: Vec<Vec<Vec<f64>>> = self
            .inputs
            .iter()
            .skip(forecast_start)
            .take(horizon)
            .map(|input_i| agent.step(input_i))
            .collect();

        println!("\nRecursive prediction of {} periods horizon", horizon);

        let y: Vec<f64> = targets.iter().map(|x| x[0][0]).collect();
        let y_hat: Vec<f64> = forecast.iter().map(|x| x[0][0]).collect();

        let error: Vec<f64> = y
            .iter()
            .zip(y_hat.iter())
            .map(|(x, x_hat)| x_hat - x)
            .collect();

        println!("Error history: {:?}", error);
    }
}

pub fn main() {
    let nb_generation: usize = 100;
    let nb_individus: usize = 1000;
    let survivial_rate: f64 = 0.02;

    let mutation_rate: f64 = 1.5;
    let mutation_decay: f64 = 0.998;

    let mut simulation = Timeserie::new();
    let mut population: Vec<NeuralNet> = (0..nb_individus).map(|_| NeuralNet::new()).collect();

    population = training_from_checkpoint::<NeuralNet, Timeserie>(
        population,
        &mut simulation,
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    let mut sim = Timeserie::new();
    let horizon: usize = 100;
    sim.test_forecast(&mut population[0], horizon, 20);
    sim.test_forecast(&mut population[0], horizon, 30);
    sim.test_forecast(&mut population[0], horizon, 40);
    sim.test_forecast(&mut population[0], horizon, 50);
    sim.test_forecast(&mut population[0], horizon, 60);

    // let best_agent_json = serde_json::to_string(&population[0]).unwrap();
    // if let Err(err) = fs::write("best_agent.json", &best_agent_json) {
    //     eprintln!("Error writing to file: {}", err);
    // } else {
    //     println!("JSON data written");
    // }

    // let best_agent_str = match fs::read_to_string("best_agent.json") {
    //     Ok(data) => data,
    //     Err(err) => {
    //         eprintln!("Error reading file: {}", err);
    //         return;
    //     }
    // };

    // let deserialized_agent: NeuralNet = serde_json::from_str(&best_agent_str).unwrap();
}
