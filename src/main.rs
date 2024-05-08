use std::cmp;
use std::iter::zip;

use anyhow::Result;

use console::{style, Emoji};
use indicatif::ProgressBar;
use libnn::activation::Activation;
use libnn::cost::Cost;
use libnn::deep_nn::DenseLayer;
use libnn::mnist_loader::MnistDataLoader;
use libnn::MNIST_TRAINING_DATA_PATHS;
use libnn::{deep_nn, simple_nn};

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍 ", "");
static TRAINING: Emoji<'_, '_> = Emoji("🏋️ ", "");
static SAVING: Emoji<'_, '_> = Emoji("💾 ", "");
static CRYSTAL_BALL: Emoji<'_, '_> = Emoji("🔮 ", "");

#[show_image::main]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        println!(
            "Usage: {} <simple|deep> <train|predict> <sample_count> <model_file>",
            args[0]
        );
        return Ok(());
    }

    let cmd = args[1].as_str();
    let subcmd = args[2].as_str();
    let sample_count = {
        let arg = args[3].as_str();
        arg.parse::<usize>().unwrap_or(0)
    };
    let model_file = args[4].as_str();
    match cmd {
        "simple" => match subcmd {
            "train" => simple_nn_train(sample_count, model_file)?,
            "predict" => simple_nn_predict(sample_count, model_file)?,
            _ => println!("Invalid command."),
        },
        "deep" => match subcmd {
            "train" => deep_nn_train(sample_count, model_file)?,
            "predict" => deep_nn_predict(sample_count, model_file)?,
            _ => println!("Invalid command."),
        },
        _ => println!("Invalid command."),
    }

    Ok(())
}

fn deep_nn_train(sample_count: usize, model_file: &str) -> Result<()> {
    let mnist = load_mnist()?;

    let (x_train, y_train) = mnist.get_training_data();

    let sample_count = cmp::min(sample_count, x_train.len());
    if sample_count > x_train.len() {
        println!(
            "Sample count ({}) exceeds training data size ({}). Using all training data.",
            sample_count,
            x_train.len()
        );
    }

    println!(
        "{} {}Training deep neural network on {} samples...",
        style("[2/3]").bold(),
        TRAINING,
        sample_count,
    );
    let pb = ProgressBar::new(sample_count as u64);

    // NN1: 92% (5k samples)
    //  - 28x28, 300; Sigmoid
    //  - 300, 10; Sigmoid
    //  - 0.1; CrossEntropy
    // NN2: 78% (5k samples)
    //  - 28x28, 300; Sigmoid
    //  - 300, 100; Sigmoid
    //  - 0.1; Quadratic
    let mut deep_nn = deep_nn::DeepNeuralNetwork::new(
        28 * 28,
        vec![
            DenseLayer::new(28 * 28, 300, Activation::Sigmoid),
            DenseLayer::new(300, 300, Activation::Sigmoid),
            DenseLayer::new(300, 10, Activation::Sigmoid),
        ],
        0.1,
        Cost::CrossEntropy,
    );

    for (x, y) in zip(x_train, y_train).take(sample_count) {
        // Normalize input data.
        let input = (x / 255.0 * 0.999) + 0.001;

        // Create target vector.
        let mut target = vec![0.001; 10];
        target[*y as usize] = 0.999;

        // Train the network.
        deep_nn.train(&input, &target);

        pb.inc(1);
    }
    pb.finish_and_clear();

    println!(
        "{} {}Saving deep neural network...",
        style("[3/3]").bold(),
        SAVING
    );
    deep_nn.save(model_file)?;

    Ok(())
}

fn deep_nn_predict(sample_count: usize, model_file: &str) -> Result<()> {
    let mnist = load_mnist()?;

    println!(
        "{} {}Loading deep neural network...",
        style("[2/3]").bold(),
        LOOKING_GLASS
    );

    let deep_nn = deep_nn::DeepNeuralNetwork::load(model_file)?;

    let (x_test, y_test) = mnist.get_test_data();

    let sample_count = cmp::min(sample_count, x_test.len());
    if sample_count > x_test.len() {
        println!(
            "Sample count ({}) exceeds test data size ({}). Using all test data.",
            sample_count,
            x_test.len()
        );
    }

    println!(
        "{} {}Predicting from {} test samples...",
        style("[3/3]").bold(),
        CRYSTAL_BALL,
        sample_count
    );

    let pb = ProgressBar::new(sample_count as u64);
    let mut correct = 0;
    for (x, y) in zip(x_test, y_test).take(sample_count) {
        let y_pred = deep_nn.predict(&x);
        if y_pred as f64 == *y {
            correct += 1;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    let accuracy = (correct as f64 / sample_count as f64) * 100.0;
    println!("Predictions finished with accuracy {:.2}%.", accuracy);

    Ok(())
}

fn simple_nn_train(sample_count: usize, model_file: &str) -> Result<()> {
    let mnist = load_mnist()?;

    let (x_train, y_train) = mnist.get_training_data();

    let sample_count = cmp::min(sample_count, x_train.len());
    if sample_count > x_train.len() {
        println!(
            "Sample count ({}) exceeds training data size ({}). Using all training data.",
            sample_count,
            x_train.len()
        );
    }

    let mut simple_nn = simple_nn::Network::new(28 * 28, 300, 10, 0.1);

    println!(
        "{} {}Training simple neural network...",
        style("[2/3]").bold(),
        TRAINING
    );
    let pb = ProgressBar::new(x_train.len() as u64);

    let target = vec![0.001; 10];
    for (x, y) in zip(x_train, y_train).take(sample_count) {
        // Normalize input data.
        let input = (x / 255.0 * 0.99) + 0.01;

        // Create target vector.
        let mut target = target.clone();
        target[*y as usize] = 0.999;

        // Train the network.
        simple_nn.train(input, target);

        // Increment progress bar.
        pb.inc(1);
    }
    pb.finish_and_clear();

    println!(
        "{} {}Saving simple neural network...",
        style("[3/3]").bold(),
        SAVING
    );
    simple_nn.save(model_file)?;

    Ok(())
}

fn simple_nn_predict(sample_count: usize, model_file: &str) -> Result<()> {
    let mnist = load_mnist()?;

    println!(
        "{} {}Loading simple neural network...",
        style("[2/3]").bold(),
        LOOKING_GLASS
    );
    let simple_nn = simple_nn::Network::load(model_file)?;

    let (x_test, y_test) = mnist.get_test_data();
    let sample_count = cmp::min(sample_count, x_test.len());
    if sample_count > x_test.len() {
        println!(
            "Sample count ({}) exceeds test data size ({}). Using all test data.",
            sample_count,
            x_test.len()
        );
    }

    println!(
        "{} {}Predicting test data...",
        style("[3/3]").bold(),
        CRYSTAL_BALL
    );
    let pb = ProgressBar::new(x_test.len() as u64);
    let mut correct = 0;
    for (x, y) in zip(x_test, y_test).take(sample_count) {
        let y_pred = simple_nn.predict(x.clone());
        if y_pred as f64 == *y {
            correct += 1;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    let accuracy = (correct as f64 / sample_count as f64) * 100.0;
    println!("Predictions finished with accuracy {:.2}%.", accuracy);

    Ok(())
}

fn load_mnist() -> Result<MnistDataLoader> {
    println!(
        "{} {}Loading MNIST dataset...",
        style("[1/3]").bold(),
        LOOKING_GLASS
    );
    let mut mnist_data_loader = MnistDataLoader::new(
        MNIST_TRAINING_DATA_PATHS[0],
        MNIST_TRAINING_DATA_PATHS[1],
        MNIST_TRAINING_DATA_PATHS[2],
        MNIST_TRAINING_DATA_PATHS[3],
        false,
    );
    mnist_data_loader.load()?;

    Ok(mnist_data_loader)
}
