use std::iter::zip;

use anyhow::Result;

use console::{style, Emoji};
use indicatif::ProgressBar;
use libnn::loader::MnistDataLoader;
use libnn::simple_nn;
use libnn::MNIST_TRAINING_DATA_PATHS;

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍 ", "");
static TRAINING: Emoji<'_, '_> = Emoji("🏋️ ", "");
static SAVING: Emoji<'_, '_> = Emoji("💾 ", "");

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
