//
///////////
///
///
///
///
///
///
///
///
///
///
///
///
///
// https://github.com/JonathanWoollett-Light/cogent/blob/master/src/layer.rs
//https://github.com/JonathanWoollett-Light/cogent/blob/master/src/neural_network.rs#L753
//https://chat.openai.com/share/e6fe503e-bfcf-4900-9f41-ce195e899a32
use std::iter::zip;

use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Cost {
    Quadratic,
}

impl Cost {
    pub fn cost(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        match self {
            Cost::Quadratic => 0.5 * (output - target).mapv(|x| x.powi(2)).sum(),
        }
    }

    pub fn derivative(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        match self {
            Cost::Quadratic => output - target,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    pub fn prime(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            }
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct DenseLayer {
    pub neuron_count: usize,
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(from: usize, neuron_count: usize, activation: Activation) -> Self {
        Self {
            neuron_count,
            w: Array2::random(
                (neuron_count, from),
                Uniform::new(-1.0 / (from as f64).sqrt(), 1.0 / (from as f64).sqrt()),
            ),
            b: Array2::random((neuron_count, 1), Uniform::new(-1.0, 1.0)),
            activation,
        }
    }

    pub fn feedforward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // z = W * a + b
        let z = self.w.dot(input) + &self.b;

        // a = σ(z)
        let a = z.mapv(|x| self.activation.activate(x));

        (a, z)
    }

    /// Backpropagate the error through the layer.
    /// Returns the error for the previous layer.
    ///
    /// # Arguments
    ///
    /// * `error` - The error of the current layer.
    /// * `a` - The output of the previous layer.
    /// * `learning_rate` - The learning rate.
    ///
    /// # Returns
    ///
    /// The partial derivative of cost for the previous layer (∂C/∂a^{l-1}).
    /// Computed from the current layer's error and the weights (W^T * δ).
    pub fn backpropagate(
        &mut self,
        partial_cost: &Array2<f64>, // ∂C/∂a
        z: &Array2<f64>,
        prev_a: &Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        // δ = ∂C/∂a * σ'(z)
        let activation_prime = z.mapv(|x| self.activation.prime(x));
        let error = partial_cost * activation_prime;

        // ∂C/∂a^{l-1} = W^T * δ
        let next_error = self.w.t().dot(&error);

        // ∂C/∂b = δ
        self.b = &self.b - (learning_rate * &error);

        // ∂C/∂W = δ * a^{l-1}
        let weight_update = &error.dot(&prev_a.t());
        self.w = &self.w - (learning_rate * weight_update);

        next_error
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeepNeuralNetwork {
    pub inputs: usize,
    pub layers: Vec<DenseLayer>,
    pub learning_rate: f64,
    pub cost: Cost,
}

impl DeepNeuralNetwork {
    pub fn new(inputs: usize, layers: Vec<DenseLayer>, learning_rate: f64, cost: Cost) -> Self {
        Self {
            inputs,
            layers,
            learning_rate,
            cost,
        }
    }

    pub fn predict(&self, input_data: &Array2<f64>) -> usize {
        let input = input_data
            .to_shape((input_data.len(), 1))
            .unwrap()
            .to_owned();
        let layer_outputs = self.feedforward(&input);
        let (output, _) = layer_outputs.last().unwrap();

        let best = output
            .iter()
            .enumerate()
            .reduce(|(best_idx, best_val), (idx, val)| {
                if val > best_val {
                    (idx, val)
                } else {
                    (best_idx, best_val)
                }
            });

        match best {
            Some((idx, _)) => idx,
            None => panic!("No best prediction found from final outputs."),
        }
    }

    pub fn train(&mut self, input_data: &Array2<f64>, target: &Vec<f64>) {
        let input = input_data
            .to_shape((input_data.len(), 1))
            .unwrap()
            .to_owned();
        let target = Array2::from_shape_vec((target.len(), 1), target.clone()).unwrap();

        // Forward pass and save the output of each layer for backpropagation.
        let mut layer_output = self.feedforward(&input);

        // Backpropagate the error through the network.
        self.backpropagate(&target, &mut layer_output);
    }

    /// Feedforward the input through the network.
    /// Returns the output of each layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to the network.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the input (a) and output pre-activation (z) of each layer.
    pub fn feedforward(&self, input: &Array2<f64>) -> Vec<(Array2<f64>, Option<Array2<f64>>)> {
        // Next input is activation from previous layer.
        let mut prev_a = input.clone();

        // Save the output of each layer for backpropagation.
        let mut layer_output = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (next_a, layer_z) = layer.feedforward(&prev_a);
            // Save the output of each layer for backpropagation.
            layer_output.push((prev_a.clone(), Some(layer_z)));
            // Update the input for the next layer (next layer input is the current layer output).
            prev_a = next_a;
        }
        layer_output.push((prev_a, None));

        layer_output
    }

    pub fn backpropagate(
        &mut self,
        target: &Array2<f64>,
        layer_output: &mut Vec<(Array2<f64>, Option<Array2<f64>>)>,
    ) {
        let mut layer_output_iter = layer_output.iter_mut().rev();
        let last_activation = &layer_output_iter.next().unwrap().0;

        let layer_iter = self.layers.iter_mut().rev();

        // ∂C/∂a
        let mut partial_cost = self.cost.derivative(last_activation, target);

        for (layer, (prev_a, z)) in zip(layer_iter, layer_output_iter) {
            partial_cost = layer.backpropagate(
                &partial_cost,
                z.as_ref().expect("No z value found for layer"),
                prev_a,
                self.learning_rate,
            );
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        bincode::serialize_into(&mut file, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let network: DeepNeuralNetwork = bincode::deserialize_from(file)?;
        Ok(network)
    }
}
