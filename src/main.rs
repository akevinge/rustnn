use std::iter::zip;

use anyhow::Result;

use console::{style, Emoji};
use indicatif::ProgressBar;
use libnn::deep_nn::{DenseLayer, QuadraticCost};
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
    if args.len() < 3 {
        println!("Usage: {} <simple|deep> <train|predict>", args[0]);
        return Ok(());
    }

    let cmd = args[1].as_str();
    let subcmd = args[2].as_str();
    match cmd {
        "simple" => match subcmd {
            "train" => simple_nn_train()?,
            "predict" => simple_nn_predict()?,
            _ => println!("Invalid command."),
        },
        "deep" => match subcmd {
            "train" => deep_nn_train()?,
            "predict" => deep_nn_predict()?,
            _ => println!("Invalid command."),
        },
        _ => println!("Invalid command."),
    }

    Ok(())
}

fn deep_nn_train() -> Result<()> {
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

    let (x_train, y_train) = mnist_data_loader.get_training_data();

    println!(
        "{} {}Training deep neural network...",
        style("[2/3]").bold(),
        TRAINING
    );
    let pb = ProgressBar::new(x_train.len() as u64);

    let mut deep_nn = deep_nn::DeepNeuralNetwork::new(
        28 * 28,
        vec![DenseLayer::new(28 * 28, 300), DenseLayer::new(300, 10)],
        0.1,
        Box::new(QuadraticCost),
    );

    for (x, y) in zip(x_train, y_train) {
        // Normalize input data.
        let input = (x / 255.0 * 0.99) + 0.01;

        // Create target vector.
        let mut target = vec![0.001; 10];
        target[*y as usize] = 0.999;

        // Train the network.
        deep_nn.train(&input, &target);

        pb.inc(1);
    }
    pb.finish_and_clear();

    Ok(())
}

fn deep_nn_predict() -> Result<()> {
    Ok(())
}

fn simple_nn_train() -> Result<()> {
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

    let (x_train, y_train) = mnist_data_loader.get_training_data();

    let mut simple_nn = simple_nn::Network::new(28 * 28, 300, 10, 0.1);

    println!(
        "{} {}Training simple neural network...",
        style("[2/3]").bold(),
        TRAINING
    );
    let pb = ProgressBar::new(x_train.len() as u64);

    let target = vec![0.001; 10];
    for (x, y) in zip(x_train, y_train) {
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
    simple_nn.save("data/simple_nn.model")?;

    Ok(())
}

fn simple_nn_predict() -> Result<()> {
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

    println!(
        "{} {}Loading simple neural network...",
        style("[2/3]").bold(),
        LOOKING_GLASS
    );
    let simple_nn = simple_nn::Network::load("data/simple_nn.model")?;

    let (x_test, y_test) = mnist_data_loader.get_test_data();

    println!(
        "{} {}Predicting test data...",
        style("[3/3]").bold(),
        CRYSTAL_BALL
    );
    let pb = ProgressBar::new(x_test.len() as u64);
    let mut correct = 0;
    for (x, y) in zip(x_test, y_test) {
        let y_pred = simple_nn.predict(x.clone());
        if y_pred as f64 == *y {
            correct += 1;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    let accuracy = (correct as f64 / y_test.len() as f64) * 100.0;
    println!("Predictions finished with accuracy {:.2}%.", accuracy);

    Ok(())
}
