mod neuralnetwork;

use genetic_rl::neuralnetwork::layer::LinearLayer;

fn main() {
    let mut layer = LinearLayer::new(3, 2);
    let w = vec![
        vec![1., 2., 3.],
        vec![3., 4., 5.]
    ];

    println!("{:?}", layer.weights);

    layer.set_weights(w, vec![0., 0.]);

    println!("{:?}", layer.weights);

    let input = vec![
        vec![1., 2., 3.],
        vec![1., 1., 1.]
    ];

    println!("{:?}", layer.forward(input));

    let layer_mut = layer.mutate(0.99);
    println!("{:?}", layer_mut.weights);
    println!("{:?}", layer_mut.bias);
}
