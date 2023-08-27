use crate::genetic_training::agent::Agent;

pub trait Simulation: Clone + Send + Sync + 'static {
    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent;

    fn on_generation(&mut self, generation_number: usize);
}
