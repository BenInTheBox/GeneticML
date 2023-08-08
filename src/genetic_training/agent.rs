pub trait Agent: Clone + Copy + Send + Sync + 'static {
    fn new() -> Self;
    fn step(&mut self, input: Vec<f64>) -> Vec<f64>;
    fn reset(&mut self);
    fn mutate(&self, mutation_rate: f64) -> Self;
}
