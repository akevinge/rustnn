use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    iter,
};

use anyhow::Result;
use ndarray::Array2;
use rand::Rng;
use show_image::{create_window, ImageInfo, ImageView};

// Loader for the MNIST dataset.
// Rust port of https://www.kaggle.com/code/hojjatk/read-mnist-dataset
pub struct MnistDataLoader {
    show_images: bool,

    x_train: Vec<Array2<f64>>,
    y_train: Vec<f64>,
    x_test: Vec<Array2<f64>>,
    y_test: Vec<f64>,

    training_data_path: String,
    training_label_path: String,
    test_data_path: String,
    test_label_path: String,
}

impl MnistDataLoader {
    pub fn new(
        training_data_path: &str,
        training_label_path: &str,
        test_data_path: &str,
        test_label_path: &str,
        show_images: bool,
    ) -> Self {
        Self {
            show_images,
            x_train: Vec::new(),
            y_train: Vec::new(),
            x_test: Vec::new(),
            y_test: Vec::new(),
            training_data_path: training_data_path.to_string(),
            training_label_path: training_label_path.to_string(),
            test_data_path: test_data_path.to_string(),
            test_label_path: test_label_path.to_string(),
        }
    }

    /// Read the MNIST dataset from the given image and label paths.
    /// The image and label paths should point to the raw MNIST dataset files.
    ///
    /// Returns a tuple containing the image data and the corresponding labels.
    fn read_data(
        &self,
        image_path: &str,
        label_path: &str,
    ) -> Result<(Vec<Array2<f64>>, Vec<f64>)> {
        const LABEL_MAGIC_NUMBER: u32 = 2049;
        const IMAGE_MAGIC_NUMBER: u32 = 2051;

        // Process label data.
        let file = File::open(label_path)?;
        let mut reader = BufReader::new(file);

        // Read and validate magic number.
        let magic_number = {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            u32::from_be_bytes(buf)
        };

        if magic_number != LABEL_MAGIC_NUMBER {
            return Err(anyhow::anyhow!(
                "Invalid magic number, expected 2049, got {}",
                magic_number
            ));
        }

        let (label_dataset_size, raw_labels) = {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            let label_dataset_size = u32::from_be_bytes(buf);

            let total_size = reader.get_ref().metadata()?.len();
            let stream_pos = reader.stream_position()?;
            let bytes_remaining = (total_size - stream_pos) as usize;
            let mut label_data = Vec::with_capacity(bytes_remaining);
            reader.read_to_end(&mut label_data)?;

            (label_dataset_size, label_data)
        };

        if label_dataset_size != raw_labels.len() as u32 {
            return Err(anyhow::anyhow!(
                "Expected label dataset size ({}) does not match read label data size ({})",
                label_dataset_size,
                raw_labels.len()
            ));
        }

        // Process image data.
        let file = File::open(image_path)?;
        let mut reader = BufReader::new(file);

        // Read and validate magic number.
        let magic_number = {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            u32::from_be_bytes(buf)
        };

        if magic_number != IMAGE_MAGIC_NUMBER {
            return Err(anyhow::anyhow!(
                "Invalid magic number, expected 2051, got {}",
                magic_number
            ));
        }

        // Read image metadata and image data.
        let (image_dataset_size, rows, cols, image_data) = {
            let mut buf = [0; 4];
            reader.read_exact(&mut buf)?;
            let image_dataset_size = u32::from_be_bytes(buf);

            reader.read_exact(&mut buf)?;
            let rows = u32::from_be_bytes(buf);

            reader.read_exact(&mut buf)?;
            let cols = u32::from_be_bytes(buf);

            let total_size = reader.get_ref().metadata()?.len();
            let stream_pos = reader.stream_position()?;
            let bytes_remaining = (total_size - stream_pos) as usize;
            let mut image_data = Vec::with_capacity(bytes_remaining);
            reader.read_to_end(&mut image_data)?;

            (image_dataset_size, rows as usize, cols as usize, image_data)
        };

        // Validate dataset sizes.
        if label_dataset_size != image_dataset_size {
            return Err(anyhow::anyhow!(
                "Label dataset size ({}) does not match image dataset size ({})",
                label_dataset_size,
                image_dataset_size
            ));
        }

        // Reshape image data into 28x28 images.
        let mut i: usize = 0;
        let raw_images: Vec<Array2<u8>> = iter::repeat_with(|| unsafe {
            let reshaped_image = Array2::from_shape_vec_unchecked(
                (28, 28),
                Vec::from(&image_data[i * rows * cols..(i + 1) * rows * cols]),
            );
            i += 1;
            reshaped_image
        })
        .take(image_dataset_size as usize)
        .collect();

        if image_dataset_size != raw_images.len() as u32 {
            return Err(anyhow::anyhow!(
                "Expected image dataset size ({}) does not match read image data size ({})",
                image_dataset_size,
                raw_images.len()
            ));
        }

        // Show 5 random images.
        if self.show_images {
            let mut rng = rand::thread_rng();
            for _ in 0..5 {
                let random_index = rng.gen_range(0..raw_images.len() as usize);
                show_image(&raw_images[random_index])?;
                println!("Label: {}", raw_labels[random_index]);
            }
        }

        // Convert raw u8 data into f64 arrays.
        let final_labels = raw_labels.into_iter().map(f64::from).collect();
        let final_images = raw_images.into_iter().map(|i| i.mapv(f64::from)).collect();

        Ok((final_images, final_labels))
    }

    pub fn load(&mut self) -> Result<()> {
        (self.x_train, self.y_train) =
            self.read_data(&self.training_data_path, &self.training_label_path)?;

        (self.x_test, self.y_test) = self.read_data(&self.test_data_path, &self.test_label_path)?;

        Ok(())
    }

    pub fn get_training_data(&self) -> (&Vec<Array2<f64>>, &Vec<f64>) {
        (&self.x_train, &self.y_train)
    }

    pub fn get_test_data(&self) -> (&Vec<Array2<f64>>, &Vec<f64>) {
        (&self.x_test, &self.y_test)
    }
}

pub fn show_image(raw_image: &Array2<u8>) -> Result<()> {
    // Convert the raw grayscale image data into an RGB image.
    let rgb_image = image::ImageBuffer::from_fn(28, 28, |j, i| {
        let pixel = raw_image[(i as usize, j as usize)];
        image::Rgb([pixel, pixel, pixel])
    });

    let image_view = ImageView::new(ImageInfo::rgb8(28, 28), &rgb_image.as_raw());
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", image_view)?;

    Ok(())
}
