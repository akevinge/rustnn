use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Relu,
}

impl Activation {
    pub fn activate(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => z.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Relu => z.mapv(|x| f64::max(0.0, x)),
        }
    }

    pub fn prime(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Sigmoid => z.mapv(|x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }),
            Activation::Relu => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
        }
    }
}
