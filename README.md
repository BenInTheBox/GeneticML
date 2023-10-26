# GeneticML

Introducing GeneticML Module, an advancement in the realm of reinforcement learning tailored for applications that defy the constraints of locally differentiable loss functions. While traditional optimization techniques often rely on gradients to navigate complex landscapes, certain scenarios, such as those encountered in highly nonlinear and discontinuous systems, render such methods ineffective. In response, our Genetic Optimization Module harnesses the power of genetic algorithms to tackle these challenges head-on. By emulating the principles of natural selection and evolution, this module offers a versatile and potent solution for optimizing complex, non-differentiable functions, revolutionizing reinforcement learning in domains where gradient-based approaches fall short. Notably, our module boasts a generic design that accommodates a wide array of problem domains, ensuring adaptability and scalability. Furthermore, its multithreaded architecture empowers efficient parallel execution, significantly accelerating the optimization process and making it an indispensable tool for real-time or resource-intensive applications.



## Main Components
The lib is designed around two main trait:

### Agent
The Agent is the object interacting with the simulation. It is the object to be optimized.
```rs
pub trait Agent: Clone + Send + Sync + 'static {
    fn step(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn reset(&mut self);
    fn mutate(&self, mutation_rate: f64) -> Self;
}
```
Step method dimensions:\
Input: $M_{batch_size \times input_size}$\
Output: $M_{batch_size \times output_size}$

### Simulation
The Simulation is the object responsible of simulating the environment and evaluating the agents.
```rs
pub trait Simulation: Clone + Send + Sync + 'static {
    fn evaluate_agent<A>(&self, agent: &mut A) -> f64
    where
        A: Agent;

    fn on_generation(&mut self, generation_number: usize);
}
```

### Neural network implementation

The agents can take an form but since it is commun to use neural networks some layers and activation functions are already implemented.

The layers available for the moment are:
- Linear
- GRU

All layers are serializable and deserializable.

And the activation functions:
- Relu
- Sigmoid
- Tanh

For more information see the xornot or timeseries forecasting example.

## Training
The training is made from a checkpoint. This checkpoint is a vector of Agent. It can be used with random agents for a training from scratch or with trained agents for transfer learning.

```rs
pub fn training_from_checkpoint<A, S>(
population: Vec<A>,
simulation: &mut S,
nb_individus: usize,
nb_generation: usize,
survivial_rate: f64,
mutation_rate: f64,
mutation_decay: f64,
    ) -> Vec<A>
    where
        A: Agent,
        S: Simulation
```

## Examples

Here are some implementation examples. These examples are only dedicated to show use to use the library. For all of them, a gradient or analytical aproach it better.

### Random Number Guesser

This is a trivial example made to understand the package.
```console
cargo run --example random_number_guess
```

Each Agent has its own guess that can't change. The Simulation contains the target number to guess and the observation is the taget with noise. The fitness is evaluated using the observation.

```rs
struct TestAgent {
    guess: f64,
}

struct TestSimulation<A: Agent> {
    agent: A,
    target: f64,
    obs: f64,
}
```

### Control system

The control system example is an inverted pendulum. The Agent is the controller and the Simulation is the inverted pendulum's simulation.

```console
cargo run --example control_system
```
![Schema](https://www.researchgate.net/profile/Kunal_Chakraborty5/publication/336134480/figure/download/fig1/AS:808758028537857@1569834348116/The-Inverted-Pendulum-System.png) 
![Equations](https://www.physicsforums.com/attachments/ss2-jpg.232719/)

With the controller equation being:
$$u = k1 * x + k2 * \dot{x} + k3 * \theta + k4 * \dot{\theta}$$
$k_{i}$ are the controller weights and also the parameters to be optimized.

```rs
pub struct Controller {
    x_coeff: f64,
    x_dot_coeff: f64,
    theta_coeff: f64,
    theta_dot_coeff: f64,
}

struct InvertedPendulum<A: Agent> {
    agent: A,
    m: f64, // Mass of the pendulum
    m_kart: f64, // Mass of the cart
    l: f64, // Length of the pendulum
    g: f64, // Acceleration due to gravity
    x: f64, // Cart position
    theta: f64, // Pendulum angle
    x_dot: f64, // Cart velocity
    theta_dot: f64, // Pendulum angular velocity
    theta_acc: f64,
    x_acc: f64,
}
```

### XorNOT

Trival example to show the use of the neural network module.

```console
cargo run --example xornot
```

### Timeseries forecasting

Timeserie forecsting example:

```console
cargo run --release --example timeseries_forecasting
```