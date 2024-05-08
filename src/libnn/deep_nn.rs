use std::iter::zip;

use anyhow::Result;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::{cost::Cost, layer::DenseLayer};

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

    pub fn predict(&self, input_data: &Array3<f64>) -> usize {
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

    pub fn train(&mut self, input_data: &Array3<f64>, target: &Vec<f64>) {
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
