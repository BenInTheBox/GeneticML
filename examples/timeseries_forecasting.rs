extern crate genetic_rl;
use plotters::prelude::*;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_scratch;
use crate::genetic_rl::neuralnetwork::activation::tanh;
use crate::genetic_rl::neuralnetwork::layer::{GRULayer, LinearLayer};
use crate::genetic_rl::neuralnetwork::metrics::calculate_mse_time_series;

#[derive(Clone)]
pub struct NeuralNet {
    layer1: GRULayer,
    layer2: GRULayer,
    layer3: LinearLayer,
}

unsafe impl Send for NeuralNet {}
unsafe impl Sync for NeuralNet {}

impl Agent for NeuralNet {
    fn new() -> Self {
        NeuralNet {
            layer1: GRULayer::new(1, 8),
            layer2: GRULayer::new(8, 8),
            layer3: LinearLayer::new(8, 1),
        }
    }

    fn reset(&mut self) {
        self.layer1.reset();
    }

    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = self.layer1.forward(input);

        output = self.layer2.forward(&output);
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

#[derive(Clone)]
struct Timeserie<A: Agent> {
    agent: A,
    pub inputs: Vec<Vec<Vec<f64>>>,
    targets: Vec<Vec<Vec<f64>>>,
}

unsafe impl<A: Agent> Send for Timeserie<A> {}
unsafe impl<A: Agent> Sync for Timeserie<A> {}

impl<A: Agent> Simulation<A> for Timeserie<A> {
    fn new(agent: A) -> Self {
        let dt: f64 = 1. / 4.;
        let amplitude: f64 = 0.3;
        let total_steps: usize = (100. / dt) as usize;

        let timeserie: Vec<f64> = (0..total_steps)
            .map(|step| amplitude * ((step as f64 * dt).sin() + (step as f64 * dt * 1.5).sin()))
            .collect();

        Timeserie {
            agent,
            inputs: timeserie
                .iter()
                .take(timeserie.len() - 1)
                .map(|x| vec![vec![*x]])
                .collect(),
            targets: timeserie.iter().skip(1).map(|x| vec![vec![*x]]).collect(),
        }
    }

    fn evaluate_agent(&mut self) -> f64 {
        self.agent.reset();
        let memory_loading_steps: usize = 50;

        self.inputs
            .iter()
            .take(memory_loading_steps)
            .for_each(|input_i| {
                let _ = self.agent.step(input_i);
            });

        let prediction: Vec<Vec<Vec<f64>>> = self
            .inputs
            .iter()
            .skip(memory_loading_steps)
            .map(|input_i| self.agent.step(input_i))
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

    fn get_agent(&self) -> A {
        self.agent.clone()
    }

    fn new_env(&self, agent: A) -> Self {
        let mut new_e = self.clone();
        new_e.agent = agent;
        new_e
    }

    fn on_generation(&mut self, _generation_number: usize) {}
}

impl<A: Agent> Timeserie<A> {
    pub fn forecast(&self, horizon: usize, current_input: &Vec<Vec<f64>>) -> Vec<Vec<Vec<f64>>> {
        let mut forecast: Vec<Vec<Vec<f64>>> = vec![];
        let mut agent = self.agent.clone();
        let mut last_prediction: Vec<Vec<f64>> = current_input.clone();

        (0..horizon).for_each(|_| {
            last_prediction = agent.step(&last_prediction);
            forecast.push(last_prediction.clone());
        });

        forecast
    }

    pub fn test_forecast(&mut self, horizon: usize, forecast_start: usize) {
        self.agent.reset();

        self.inputs.iter().take(forecast_start).for_each(|input_i| {
            let _ = self.agent.step(input_i);
        });

        let forecast = self.forecast(horizon, &self.inputs[forecast_start + 1]);

        let targets: Vec<Vec<Vec<f64>>> = self
            .inputs
            .iter()
            .skip(forecast_start)
            .take(horizon)
            .map(|input_i| self.agent.step(input_i))
            .collect();

        println!("\nRecursive prediction of {} periods horizon", horizon);

        let y: Vec<(f32, f32)> = targets
            .iter()
            .enumerate()
            .map(|(i, x)| (i as f32, x[0][0] as f32))
            .collect();
        let y_hat: Vec<(f32, f32)> = forecast
            .iter()
            .enumerate()
            .map(|(i, x)| (i as f32, x[0][0] as f32))
            .collect();
        let name = format!("forecast_start_{}.png", forecast_start);

        let root_area = BitMapBackend::new(&name, (1024, 768)).into_drawing_area();

        root_area.fill(&WHITE).unwrap();

        let root_area = root_area.titled("Backtest", ("sans-serif", 60)).unwrap();

        let (upper, _lower) = root_area.split_vertically(512);

        let mut cc = ChartBuilder::on(&upper)
            .margin(5)
            .set_all_label_area_size(50)
            .caption("Target vs Forecast", ("sans-serif", 40))
            .build_cartesian_2d(0f32..(y.len() as f32), -0.6f32..0.6f32)
            .unwrap();

        cc.configure_mesh()
            .x_labels(20)
            .y_labels(10)
            .disable_mesh()
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .draw()
            .unwrap();

        cc.draw_series(LineSeries::new(y, &RED))
            .unwrap()
            .label("Target")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        cc.draw_series(LineSeries::new(y_hat, &BLUE))
            .unwrap()
            .label("Prediction")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        cc.configure_series_labels()
            .border_style(&BLACK)
            .draw()
            .unwrap();
    }
}

pub fn main() {
    let nb_generation: usize = 1000;
    let nb_individus: usize = 1000;
    let survivial_rate: f64 = 0.02;

    let mutation_rate: f64 = 1.5;
    let mutation_decay: f64 = 0.997;

    let population = training_from_scratch::<NeuralNet, Timeserie<NeuralNet>>(
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    let mut sim = Timeserie::new(population[0].clone());
    let horizon: usize = 100;
    sim.test_forecast(horizon, 20);
    sim.test_forecast(horizon, 30);
    sim.test_forecast(horizon, 40);
    sim.test_forecast(horizon, 50);
    sim.test_forecast(horizon, 60);
}
