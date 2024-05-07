use anyhow::Result;
use ndarray::{Array2, CowArray, Dim};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use serde::{Deserialize, Serialize};

type SingleColumnArray<'a> = CowArray<'a, f64, Dim<[usize; 2]>>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Network {
    pub inputs: usize,
    pub hiddens: usize,
    pub outputs: usize,
    pub hidden_weights: Array2<f64>,
    pub output_weights: Array2<f64>,
    pub learning_rate: f64,
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Network {
    pub fn new(inputs: usize, hiddens: usize, outputs: usize, learning_rate: f64) -> Self {
        Self {
            inputs,
            hiddens,
            outputs,
            learning_rate,
            hidden_weights: Array2::random(
                (hiddens, inputs), // Dimensions adjusted for order of dot product: Weights (h, i) * input (i, 1) = hidden (h, 1)
                Uniform::new(-1.0 / (inputs as f64).sqrt(), 1.0 / (inputs as f64).sqrt()),
            ),
            output_weights: Array2::random(
                (outputs, hiddens), // Dimensions adjusted for order of dot product: Weights (o, h) * hidden (h, 1) = output (o, 1)
                Uniform::new(
                    -1.0 / (hiddens as f64).sqrt(),
                    1.0 / (hiddens as f64).sqrt(),
                ),
            ),
        }
    }

    pub fn forward_pass(&self, input: &SingleColumnArray) -> (Array2<f64>, Array2<f64>) {
        // a¹ = σ(W * a⁰ + b)
        let hidden_activations = self.hidden_weights.dot(input).mapv(sigmoid);

        // o = σ(W * a¹ + b)
        let final_outputs = self.output_weights.dot(&hidden_activations).mapv(sigmoid);
        (hidden_activations, final_outputs)
    }

    pub fn predict(&self, input_data: Array2<f64>) -> usize {
        let input: SingleColumnArray = input_data.to_shape((input_data.len(), 1)).unwrap();
        let (_, final_outputs) = self.forward_pass(&input);

        let best = final_outputs
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

    pub fn train(&mut self, input_data: Array2<f64>, target_data: Vec<f64>) {
        let input: SingleColumnArray = input_data.to_shape((input_data.len(), 1)).unwrap();
        let (hidden_outputs, final_outputs) = self.forward_pass(&input);

        let targets = Array2::from_shape_vec((target_data.len(), 1), target_data).unwrap();
        let final_errors = &targets - &final_outputs;
        let hidden_errors = &self.output_weights.t().dot(&final_errors);

        self.output_weights = &self.output_weights
            + (self.learning_rate
                * (final_errors * final_outputs.mapv(|x| x * (1.0 - x))).dot(&hidden_outputs.t()));

        self.hidden_weights = &self.hidden_weights
            + (self.learning_rate
                * (hidden_errors * hidden_outputs.mapv(|x| x * (1.0 - x))).dot(&input.t()));
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        bincode::serialize_into(&mut file, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Ok(bincode::deserialize_from(&mut file)?)
    }
}
