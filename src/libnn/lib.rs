pub mod loader;
pub mod simple_nn;

pub const MNIST_TRAINING_DATA_PATHS: [&str; 4] = [
    "data/mnist/train-images-idx3-ubyte",
    "data/mnist/train-labels-idx1-ubyte",
    "data/mnist/t10k-images-idx3-ubyte",
    "data/mnist/t10k-labels-idx1-ubyte",
];

#[cfg(test)]
mod test;
