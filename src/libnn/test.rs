use ndarray::array;

use crate::{activation::Activation, cost::Cost};

use self::deep_nn::DenseLayer;

use super::*;

#[test]
fn test_simple_forward_pass() {
    let network = simple_nn::Network {
        inputs: 2,
        hiddens: 2,
        outputs: 1,
        hidden_weights: array![[0.1, 0.4], [0.2, 0.5]],
        output_weights: array![[0.3, 0.6]],
        learning_rate: 0.1,
    };
    let input_data = array![[0.0], [1.0]];
    let input = input_data.to_shape((2, 1)).unwrap();

    let (hidden_activations, final_outputs) = network.forward_pass(&input);

    assert_eq!(
        hidden_activations,
        array![[0.598687660112452], [0.6224593312018546]],
    );
    assert_eq!(final_outputs, array![[0.6348503185851494]]);
}

#[test]
fn test_deep_forward_pass() {
    let nn = deep_nn::DeepNeuralNetwork {
        inputs: 2,
        layers: vec![
            DenseLayer {
                neuron_count: 2,
                b: array![[0.1], [0.2]],
                w: array![[0.1, 0.4], [0.2, 0.5]],
                activation: Activation::Sigmoid,
            },
            DenseLayer {
                neuron_count: 1,
                b: array![[0.3]],
                w: array![[0.3, 0.6]],
                activation: Activation::Sigmoid,
            },
        ],
        learning_rate: 0.1,
        cost: Cost::Quadratic,
    };

    let input_data = array![[0.0], [1.0]];
    let input = input_data
        .to_shape((input_data.len(), 1))
        .unwrap()
        .to_owned();
    let layer_outs = nn.feedforward(&input);

    assert_eq!(
        layer_outs,
        vec![
            (array![[0.0], [1.0]], Some(array![[0.5], [0.7]])),
            (
                array![[0.6224593312018546], [0.6681877721681662]],
                Some(array![[0.8876504626614561]])
            ),
            (array![[0.7084050726764932]], None),
        ]
    );
}
