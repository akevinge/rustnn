use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Cost {
    Quadratic,
    CrossEntropy,
}

impl Cost {
    pub fn cost(&self, output: &Array2<f64>, target: &Array2<f64>) -> f64 {
        match self {
            Cost::Quadratic => 0.5 * (output - target).mapv(|x| x.powi(2)).sum(),
            Cost::CrossEntropy => {
                let mut sum = 0.0;
                for (o, t) in output.iter().zip(target.iter()) {
                    sum += t * o.ln() + (1.0 - t) * (1.0 - o).ln();
                }
                -sum
            }
        }
    }

    pub fn derivative(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        match self {
            Cost::Quadratic => output - target,
            Cost::CrossEntropy => {
                // -y/a + (1-y)/(1-a)
                return (-target / output) + (1.0 - target) / (1.0 - output);
            }
        }
    }
}
