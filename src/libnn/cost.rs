use ndarray::Array2;
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
