use crate::genetic_training::agent::Agent;

pub trait Simulation<A: Agent>: Clone + Send + Sync + 'static {
    fn new(agent: A) -> Self;
    fn evaluate_agent(&mut self) -> f64;
    fn get_agent(&self) -> A;
    fn new_env(&self, agent: A) -> Self;
    fn on_generation(&mut self, generation_number: usize);
}
