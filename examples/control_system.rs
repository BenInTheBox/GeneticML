extern crate genetic_rl;

use crate::genetic_rl::genetic_training::agent::Agent;
use crate::genetic_rl::genetic_training::simulation::Simulation;
use crate::genetic_rl::genetic_training::training::training_from_scratch;

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
struct InvertedPendulum<A: Agent> {
    agent: A,
    m: f64,         // Mass of the pendulum
    m_kart: f64,    // Mass of the cart
    l: f64,         // Length of the pendulum
    g: f64,         // Acceleration due to gravity
    x: f64,         // Cart position
    theta: f64,     // Pendulum angle
    x_dot: f64,     // Cart velocity
    theta_dot: f64, // Pendulum angular velocity
    theta_acc: f64,
    x_acc: f64,
}

unsafe impl<A: Agent> Send for InvertedPendulum<A> {}
unsafe impl<A: Agent> Sync for InvertedPendulum<A> {}

impl<A: Agent> Simulation<A> for InvertedPendulum<A> {
    fn new(agent: A) -> Self {
        let starting_angle = rand::thread_rng().gen_range(-MAX_STARTING_ANGLE..=MAX_STARTING_ANGLE);
        println!("Starting_angle: {:.2} deg", starting_angle.to_degrees());
        InvertedPendulum {
            agent,
            m: 0.25,               // Mass of the pendulum
            m_kart: 0.25,          // Mass of the cart
            l: 0.1,                // Length of the pendulum
            g: 9.81,               // Acceleration due to gravity
            x: 0.,                 // Cart position
            theta: starting_angle, // Pendulum angle
            x_dot: 0.,             // Cart velocity
            theta_dot: 0.,         // Pendulum angular velocity
            x_acc: 0.,
            theta_acc: 0.,
        }
    }

    fn evaluate_agent(&mut self) -> f64 {
        self.simulate_agent(true)
    }

    fn get_agent(&self) -> A {
        self.agent.clone()
    }

    fn new_env(&self, agent: A) -> Self {
        InvertedPendulum {
            agent,
            m: 0.25,           // Mass of the pendulum
            m_kart: 0.25,      // Mass of the cart
            l: 0.1,            // Length of the pendulum
            g: 9.81,           // Acceleration due to gravity
            x: 0.,             // Cart position
            theta: self.theta, // Pendulum angle
            x_dot: 0.,         // Cart velocity
            theta_dot: 0.,     // Pendulum angular velocity
            x_acc: 0.,
            theta_acc: 0.,
        }
    }

    fn on_generation(&mut self, _generation_number: usize) {
        let starting_angle = rand::thread_rng().gen_range(-MAX_STARTING_ANGLE..=MAX_STARTING_ANGLE);
        self.theta = starting_angle;
    }
}

impl<A: Agent> InvertedPendulum<A> {
    fn update(&mut self, a: f64) {
        let x_acc = (self.m * self.l * self.theta_dot.powi(2) * f64::sin(self.theta)
            - self.m * self.g * f64::cos(self.theta) * f64::sin(self.theta))
            / (self.m_kart + self.m * (1.0 - f64::cos(self.theta).powi(2)));
        let theta_acc = (self.g * f64::sin(self.theta)
            + f64::cos(self.theta)
                * (-a - self.m * self.l * self.theta_dot.powi(2) * f64::sin(self.theta)))
            / (self.l * (self.m_kart + self.m * (1.0 - f64::cos(self.theta).powi(2))));

        self.x_acc = x_acc;
        self.theta_acc = theta_acc;
        self.x_dot += x_acc * DT;
        self.theta_dot += theta_acc * DT;
        self.x += self.x_dot * DT;
        self.theta += self.theta_dot * DT;
    }

    fn simulate_agent(&mut self, training: bool) -> f64 {
        let mut agent = self.agent.clone();

        let total_steps = (TOTAL_TIME / DT) as usize;

        let mut cum_squared_u = 0.;
        let mut cum_squared_error_x = 0.;
        let mut cum_squared_error_theta = 0.;

        for step in 0..total_steps {
            let x_error = 0. - self.x;
            let theta_error = 0. - self.theta;

            cum_squared_error_x += x_error.powf(2.);
            cum_squared_error_theta += theta_error.powf(2.);

            let a = agent.step(&vec![
                vec![
                    self.x * 10.,
                    self.x_dot,
                    self.theta * 3.,
                    self.theta_dot,
                ]
            ])[0][0];
            cum_squared_u += a.powf(2.);
            self.update(a);

            // Print the cart position and pendulum angle
            if !training && step % 10 == 0 {
                println!(
                    "Time: {:.2}s, Cart Position: {:.2}m, Cart Speed: {:.2}m/s, Pendulum Angle: {:.2}deg, U: {:.2}m/s²",
                    step as f64 * DT,
                    self.x,
                    self.x_dot,
                    self.theta.to_degrees(),
                    a
                );
            }

            if self.theta.abs() > PI / 2. || self.x_dot.abs() > 25. {
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
    let nb_generation: usize = 10;
    let nb_individus: usize = 100;
    let survivial_rate: f64 = 0.1;

    let mutation_rate: f64 = 0.1;
    let mutation_decay: f64 = 0.999;

    let population = training_from_scratch::<Controller, InvertedPendulum<Controller>>(
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    );

    let mut sim = InvertedPendulum::<Controller>::new(population[0]);

    println!("\n\n\n");
    sim.simulate_agent(false);
}
