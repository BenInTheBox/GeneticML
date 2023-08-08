extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_scratch;
use rand::Rng;

#[derive(Clone, Copy)]
struct TestAgent {
    guess: f64,
}

unsafe impl Send for TestAgent {}
unsafe impl Sync for TestAgent {}

impl Agent for TestAgent {
    fn new() -> Self {
        let random_number = rand::thread_rng().gen_range(-100.0..=100.0) as f64;
        TestAgent {
            guess: random_number,
        }
    }

    fn step(&mut self, _input: Vec<f64>) -> Vec<f64> {
        //println!("Guess: {}", self.guess);
        vec![self.guess]
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

#[derive(Clone)]
struct TestSimulation<A: Agent> {
    agent: A,
    target: f64,
    obs: f64,
}

unsafe impl<A: Agent> Send for TestSimulation<A> {}
unsafe impl<A: Agent> Sync for TestSimulation<A> {}

impl<A: Agent> Simulation<A> for TestSimulation<A> {
    fn new(agent: A) -> Self {
        let random_number = rand::thread_rng().gen_range(-100.0..=100.0);
        println!("Number to guess: {}", random_number);
        TestSimulation {
            agent,
            target: random_number,
            obs: random_number,
        }
    }

    fn evaluate_agent(&mut self) -> f64 {
        let mut agent = self.agent;
        let fitness = -(agent.step(vec![0.])[0] - self.obs).abs();

        fitness
    }

    fn get_agent(&self) -> A {
        self.agent
    }

    fn new_env(&self, agent: A) -> Self {
        TestSimulation {
            agent,
            target: self.target,
            obs: self.obs,
        }
    }

    fn on_generation(&mut self, generation_number: usize) {
        let ampl = 1. / (generation_number + 1) as f64;
        let random_number = rand::thread_rng().gen_range(-ampl..=ampl);
        self.obs = self.target + random_number
    }
}

pub fn main() {
    let nb_generation: usize = 10;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.1;

    let mutation_rate: f64 = 1.;
    let mutation_decay: f64 = 0.99;

    let population = training_from_scratch::<TestAgent, TestSimulation<TestAgent>>(
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    println!("Final guess: {}", population[0].to_owned().step(vec![])[0]);
}
