pub trait Agent: Clone + Send + Sync + 'static {
    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn reset(&mut self);
    fn mutate(&self, mutation_rate: f64) -> Self;
}
