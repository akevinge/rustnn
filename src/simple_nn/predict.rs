use std::iter::zip;

use anyhow::Result;

use console::style;
use console::Emoji;
use indicatif::ProgressBar;
use libnn::mnist_loader::MnistDataLoader;
use libnn::simple_nn;
use libnn::MNIST_TRAINING_DATA_PATHS;

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍 ", "");
static CRYSTAL_BALL: Emoji<'_, '_> = Emoji("🔮 ", "");

#[show_image::main]
fn main() -> Result<()> {
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
