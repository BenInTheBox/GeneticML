mod examples;
mod genetic_training;

use examples::control_systems::inverted_pendulum_exemple;
use examples::random_number_guess::random_number_guess_exemple;

use rand::Rng;
use std::sync::{Arc, Mutex};


fn main() {
    inverted_pendulum_exemple();
}