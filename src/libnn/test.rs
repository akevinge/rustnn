use ndarray::array;

use super::*;

#[test]
fn test_forward_pass() {
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
