extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_checkpoint;

use rand::Rng;
use std::f64::consts::PI;

const DT: f64 = 0.01; // Time step for simulation
const TOTAL_TIME: f64 = 3.0; // Total simulation time
const MAX_U: f64 = 9.; // Max kart acceleration

const MAX_STARTING_ANGLE: f64 = PI / 4.; // Maximum pole init*ial angle

#[derive(Clone, Copy)]
pub struct Controller {
    x_coeff: f64,
    x_dot_coeff: f64,
    theta_coeff: f64,
    theta_dot_coeff: f64,
}

unsafe impl Send for Controller {}
unsafe impl Sync for Controller {}

impl Agent for Controller {
    fn new() -> Self {
        let x_coeff = rand::thread_rng().gen_range(-1.0..=1.0);
        let x_dot_coeff = rand::thread_rng().gen_range(-1.0..=1.0);
        let theta_coeff = rand::thread_rng().gen_range(-1.0..=1.0);
        let theta_dot_coeff = rand::thread_rng().gen_range(-1.0..=1.0);
        Controller {
            x_coeff,
            x_dot_coeff,
            theta_coeff,
            theta_dot_coeff,
        }
    }

    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut a = (self.x_coeff * input[0][0])
            + (self.x_dot_coeff * input[0][1])
            + (self.theta_coeff * input[0][2])
            + (self.theta_dot_coeff * input[0][3]);
        a = f64::max(-MAX_U, f64::min(a, MAX_U));
        vec![vec![a]]
    }

    fn reset(&mut self) {}

    fn mutate(&self, mutation_rate: f64) -> Self {
        let x_coeff_delta = rand::thread_rng().gen_range(-mutation_rate..=mutation_rate);
        let x_dot_coeff_delta = rand::thread_rng().gen_range(-mutation_rate..=mutation_rate);
        let theta_dot_coeff_delta = rand::thread_rng().gen_range(-mutation_rate..=mutation_rate);
        let theta_coeff_delta = rand::thread_rng().gen_range(-mutation_rate..=mutation_rate);

        Controller {
            x_coeff: self.x_coeff + x_coeff_delta,
            x_dot_coeff: self.x_dot_coeff + x_dot_coeff_delta,
            theta_coeff: self.theta_coeff + theta_coeff_delta,
            theta_dot_coeff: self.theta_dot_coeff + theta_dot_coeff_delta,
        }
    }
}

#[derive(Clone)]
struct InvertedPendulum {
    m: f64,         // Mass of the pendulum
    m_kart: f64,    // Mass of the cart
    l: f64,         // Length of the pendulum
    g: f64,         // Acceleration due to gravity
    starting_angle: f64,
}

impl Simulation for InvertedPendulum {

    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent,
    {
        self.simulate_agent(agent, true)
    }

    fn on_generation(&mut self, _generation_number: usize) {
        let starting_angle = rand::thread_rng().gen_range(-MAX_STARTING_ANGLE..=MAX_STARTING_ANGLE);
        self.starting_angle = starting_angle;
    }
}

impl InvertedPendulum {

    fn new() -> Self {
        let starting_angle = rand::thread_rng().gen_range(-MAX_STARTING_ANGLE..=MAX_STARTING_ANGLE);
        println!("Starting_angle: {:.2} deg", starting_angle.to_degrees());
        InvertedPendulum {
            m: 0.25,               // Mass of the pendulum
            m_kart: 0.25,          // Mass of the cart
            l: 0.1,                // Length of the pendulum
            g: 9.81,               // Acceleration due to gravity
            starting_angle,
        }
    }

    fn simulate_agent<A>(&self, agent: &mut A, training: bool) -> f64
    where
        A: Agent,
    {
        let total_steps = (TOTAL_TIME / DT) as usize;

        let mut cum_squared_u = 0.;
        let mut cum_squared_error_x = 0.;
        let mut cum_squared_error_theta = 0.;

        let mut x: f64 = 0.;
        let mut theta: f64 = self.starting_angle;
        let mut x_dot: f64 = 0.;
        let mut theta_dot: f64 = 0.;
        let mut theta_acc: f64 = 0.;
        let mut x_acc: f64 = 0.;

        for step in 0..total_steps {
            let x_error = 0. - x;
            let theta_error = 0. - theta;

            cum_squared_error_x += x_error.powf(2.);
            cum_squared_error_theta += theta_error.powf(2.);

            let a = agent.step(&vec![vec![
                x * 10.,
                x_dot,
                theta * 3.,
                theta_dot,
            ]])[0][0];
            cum_squared_u += a.powf(2.);
            
            // Update state
            x_acc = (self.m * self.l * theta_dot.powi(2) * f64::sin(theta)
            - self.m * self.g * f64::cos(theta) * f64::sin(theta))
            / (self.m_kart + self.m * (1.0 - f64::cos(theta).powi(2)));

            theta_acc = (self.g * f64::sin(theta)
                + f64::cos(theta)
                    * (-a - self.m * self.l * theta_dot.powi(2) * f64::sin(theta)))
                / (self.l * (self.m_kart + self.m * (1.0 - f64::cos(theta).powi(2))));

            x_dot += x_acc * DT;
            theta_dot += theta_acc * DT;
            x += x_dot * DT;
            theta += theta_dot * DT;

            // Print the cart position and pendulum angle
            if !training && step % 10 == 0 {
                println!(
                    "Time: {:.2}s, Cart Position: {:.2}m, Cart Speed: {:.2}m/s, Pendulum Angle: {:.2}deg, U: {:.2}m/s²",
                    step as f64 * DT,
                    x,
                    x_dot,
                    theta.to_degrees(),
                    a
                );
            }

            if theta.abs() > PI / 2. || x_dot.abs() > 25. {
                return step as f64;
            }
        }

        10000.
            - (cum_squared_error_x / total_steps as f64).powf(0.5)
            - (cum_squared_error_theta / total_steps as f64).powf(0.5)
            - (cum_squared_u / total_steps as f64).powf(0.5)
    }
}

pub fn main() {
    let nb_generation: usize = 100;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.1;

    let mutation_rate: f64 = 0.1;
    let mutation_decay: f64 = 0.999;

    let mut simulation = InvertedPendulum::new();
    let mut population: Vec<Controller> = (0..nb_individus).map(|_| Controller::new()).collect();

    population = training_from_checkpoint::<Controller, InvertedPendulum>(
        population,
        &mut simulation,
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    println!("\n\n\n");
    simulation.simulate_agent(&mut population[0], false);
}
