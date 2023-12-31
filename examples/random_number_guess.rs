extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_checkpoint;
use rand::Rng;

#[derive(Clone)]
struct TestAgent {
    guess: f64,
}

unsafe impl Send for TestAgent {}
unsafe impl Sync for TestAgent {}

impl Agent for TestAgent {

    fn step(&mut self, _input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        //println!("Guess: {}", self.guess);
        vec![vec![self.guess]]
    }

    fn reset(&mut self) {}

    fn mutate(&self, mutation_rate: f64) -> Self {
        let amplitude = mutation_rate * 10.;
        let random_number = rand::thread_rng().gen_range(-amplitude..=amplitude);
        TestAgent {
            guess: self.guess + random_number,
        }
    }
}

impl TestAgent {
    fn new() -> Self {
        let random_number = rand::thread_rng().gen_range(-100.0..=100.0) as f64;
        TestAgent {
            guess: random_number,
        }
    }
}

#[derive(Clone)]
struct TestSimulation {
    pub target: f64,
    obs: f64,
}

impl Simulation for TestSimulation {
    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent,
    {
        let mut agent = agent.clone();
        let fitness = -(agent.step(&vec![])[0][0] - self.obs).abs();

        fitness
    }

    fn on_generation(&mut self, generation_number: usize) {
        let ampl = 1. / (generation_number + 1) as f64;
        let random_number = rand::thread_rng().gen_range(-ampl..=ampl);
        self.obs = self.target + random_number;
        println!("Generation observation: {}", self.obs);
    }
}

impl TestSimulation {
    fn new() -> Self {
        let random_number = rand::thread_rng().gen_range(-100.0..=100.0);
        println!("Number to guess: {}", random_number);
        TestSimulation {
            target: random_number,
            obs: random_number,
        }
    }
}

pub fn main() {
    let nb_generation: usize = 10;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.1;

    let mutation_rate: f64 = 1.;
    let mutation_decay: f64 = 0.99;

    let mut simulation = TestSimulation::new();

    let mut population: Vec<TestAgent> = (0..nb_individus).map(|_| TestAgent::new()).collect();

    population = training_from_checkpoint::<TestAgent, TestSimulation>(
        population,
        &mut simulation,
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    println!(
        "\n\nFinal guess: {}\nTarget: {}",
        population[0].to_owned().step(&vec![])[0][0], simulation.target
    );
}
