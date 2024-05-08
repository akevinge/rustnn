use serde::{Deserialize, Serialize};

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
