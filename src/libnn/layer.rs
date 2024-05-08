use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;

pub enum Layer {
    Dense(DenseLayer),
    // Conv(ConvLayer),
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
        let a = self.activation.activate(&z);

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
        let activation_prime = self.activation.prime(z);
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
