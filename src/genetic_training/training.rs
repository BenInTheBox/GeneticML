use crate::genetic_training::agent::Agent;
use crate::genetic_training::simulation::Simulation;

use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

fn run_generation<A, S>(population: Vec<A>, simulation: Arc<S>) -> Vec<(A, f64)>
where
    A: Agent,
    S: Simulation,
{
    let results: Vec<(A, f64)> = population
        .par_iter()
        .map(|agent| {
            let fitness = simulation.evaluate_agent(&mut agent.clone());
            (agent.clone(), fitness) // Clone agent for the result tuple
        })
        .collect();

    let mut results = results; // Move results to sort them

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    results
}

fn reproduce<A>(mut population: Vec<A>, nb_individus: usize, mutation_rate: f64) -> Vec<A>
where
    A: Agent,
{
    let pop = population.len();
    (0..(nb_individus - pop))
        .for_each(|i| population.push(population[i % pop].mutate(mutation_rate)));
    for agent in &mut population {
        agent.reset();
    }
    population
}

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
    S: Simulation,
{
    let nb_keep: usize = (nb_individus as f64 * survivial_rate) as usize;

    let mut population: Vec<A> = reproduce(population, nb_individus * 100, mutation_rate);

    let s_time = Instant::now();

    for gen in 0..nb_generation {
        let start_time = Instant::now();
        let gen_mutation_rate = mutation_rate * mutation_decay.powf(gen as f64);
        println!(
            "Generation: {}     Mutation rate: {}",
            gen,
            (gen_mutation_rate * 10000.0).round() / 10000.0
        );
        let sim = Arc::new(simulation.clone());

        let results = run_generation(population, sim);

        let surviviors: Vec<(A, f64)> = results.into_iter().take(nb_keep).collect();

        let scores: Vec<f64> = surviviors
            .iter()
            .map(|res| (res.1 * 10000.0).round() / 10000.0)
            .collect();
        println!("Best individuals fitness: {:?}", scores);

        population = reproduce(
            surviviors.into_iter().map(|sur| sur.0).collect(),
            nb_individus,
            gen_mutation_rate,
        );

        simulation.on_generation(gen);

        let end_time = Instant::now();
        let elapsed_time = end_time.duration_since(start_time);
        let elapsed_seconds = elapsed_time.as_millis();

        println!("{} ms\n", elapsed_seconds);
    }

    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(s_time);
    let elapsed_seconds = elapsed_time.as_millis();
    println!(
        "Total time: {} ms\nFor {} individuals for {} generations.\nFor a total of {} simulations",
        elapsed_seconds,
        nb_individus,
        nb_generation,
        nb_individus * nb_generation
    );

    population
}

/* 
pub fn training_from_scratch<A, S>(
    nb_individus: usize,
    nb_generation: usize,
    survivial_rate: f64,
    mutation_rate: f64,
    mutation_decay: f64,
) -> Vec<A>
where
    A: Agent,
    S: Simulation<A>,
{
    let model_agent = A::new();
    let mut sim = S::new(model_agent);

    let population: Vec<A> = (0..nb_individus).map(|_| A::new()).collect();

    training_from_checkpoint::<A, S>(
        population,
        &mut sim,
        nb_individus,
        nb_generation,
        survivial_rate,
        mutation_rate,
        mutation_decay,
    )
}
*/