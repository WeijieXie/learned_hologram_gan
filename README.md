# Watermelon Project

**Hi, there!**

This project trains a neural network to generate phase-only holograms from the input of RGBD images. If you're unfamiliar with the background concepts behind holography and deep learning, there's a folder called [`warmingUp`](warmingUp) with Jupyter notebooks that contain basic information. Feel free to check it out if you're interested.

**NOTICE: The pipeline of this project is RAM-intensive. To generate 4K holograms, you will need at least 30GB of RAM (such as in an A100 GPU). For smaller 392x392 holograms, a personal computer should suffice.**

## How to Set Up the Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/WeijieXie/manuscript_watermelon.git
   ```

2. **Install the required packages**:

   Navigate to the cloned folder and run the following:

   ```bash
   pip install .
   ```

   You may also need to install the following additional packages:

   ```bash
   pip install Imath openexr torchmetrics
   ```

3. **Prepare the dataset**:

   You can download the .bin version of the MIT-CGH-4K dataset [here: MIT-CGH-4K-BIN](https://drive.google.com/drive/folders/1cY4B12Rvds-kx5wplE7J2zziAJiuNc2N?usp=drive_link).

   Alternatively, if you have the .exr version of the dataset, you can convert it to .bin format by running the following command:

   ```bash
   python exr2bin.py data/test_192 --channelsNum 3 --height 192 --width 192 
   ```

   Replace `data/test_192` with the path to the folder containing subfolders of `.exr` files for the RGBD images. The `--channelsNum` argument refers to the number of channels in the RGB images, and the `--height` and `--width` arguments specify the dimensions of the RGBD images.

## How to Train the Model

To train the model, you can use the provided `trainingModel.py` script. Below is an example of how to run the training script with various parameters.

### Example Training Command

```bash
python trainingModel.py --train_img_path data/train_384/img.bin \
--train_depth_path data/train_384/depth.bin \
--train_amp_path data/train_384/amp.bin \
--train_phs_path data/train_384/phs.bin \
--validate_img_path data/validate_384/img.bin \
--validate_depth_path data/validate_384/depth.bin \
--validate_amp_path data/validate_384/amp.bin \
--validate_phs_path data/validate_384/phs.bin \
--samplesNum 500 \
--channlesNum 3 \
--height 384 \
--width 384 \
--batch_size 4 \
--lr_G 1e-3 \
--lr_D 1e-3 \
--epoch_num 50 \
--save_path_G output/models/generator.pth \
--save_path_D output/models/discriminator.pth \
--loss_metrics_file output/metrics/loss_metrics.csv \
--save_path_img output/images
```

### Description of Arguments

- `--train_img_path`: Path to the training image binary file.
- `--train_depth_path`: Path to the training depth binary file.
- `--train_amp_path`: Path to the training amplitude binary file.
- `--train_phs_path`: Path to the training phase binary file.
- `--validate_img_path`: Path to the validation image binary file.
- `--validate_depth_path`: Path to the validation depth binary file.
- `--validate_amp_path`: Path to the validation amplitude binary file.
- `--validate_phs_path`: Path to the validation phase binary file.
- `--samplesNum`: Number of samples in the dataset.
- `--channlesNum`: Number of channels in the image (typically 3 for RGB).
- `--height`: Image height.
- `--width`: Image width.
- `--batch_size`: Batch size for training.
- `--lr_G`: Learning rate for the generator.
- `--lr_D`: Learning rate for the discriminator.
- `--epoch_num`: Number of epochs for training.
- `--save_path_G`: Path to save the trained generator model.
- `--save_path_D`: Path to save the trained discriminator model.
- `--loss_metrics_file`: Path to save the loss metrics during training.
- `--save_path_img`: Path to save generated images during training.

## How to Generate Holograms with the Trained Model

To generate holograms using a pre-trained model, you can use the `generatePOH.py` script. The script will take an RGBD image as input and generate a phase-only hologram (POH). Optionally, it can also perform holographic propagation. [Here](data/test_384) are some rgb and depth images from the test dataset which are unseen by the model and can be used to infer the model's performance. The model is also available [here](output/models).

### Example Command to Generate a POH

```bash
python generatePOH.py --img_path data/test_384/img.bin \
--depth_path data/test_384/depth.bin \
--index 99 \
--model_path output/models/watermelon_GAN_GENERATOR_E_384_COMPARISON.pth \
--poh_output_path output/test_output/terminalTest/poh.pt \
--propagate \
--num_intervals 10 \
--output_image_dir output/test_output/terminalTest
```

### Description of Arguments

- `--img_path`: Path to the input image binary file.
- `--depth_path`: Path to the input depth binary file.
- `--index`: Index of the sample in the dataset to generate the POH for.
- `--model_path`: Path to the pre-trained generator model.
- `--poh_output_path`: Path to save the generated POH file.
- `--propagate`: Flag to enable propagation after generating the POH.
- `--num_intervals`: Number of intervals for the propagation distances.
- `--output_image_dir`: Directory to save the propagated images.

### Propagation Process

If the `--propagate` flag is set, the script will propagate the generated POH over several distances (controlled by `--min_distance`, `--max_distance`, and `--num_intervals`). The result of the propagation will be saved as images in the specified `--output_image_dir`.

### Example Generated Output

When you run the command provided above, the following things will happen:

1. A POH will be generated from the specified RGBD image (in this case, index 99 from the dataset).
2. The generated POH will be saved to the path `output/test_output/terminalTest/poh.pt`.
3. The script will perform propagation over a series of distances (as defined by `--num_intervals 10`).
4. The resulting propagated images will be saved in the `output/test_output/terminalTest` directory.

### Additional Parameters for Hologram Generation

- `--samplesNum`: Number of samples in the dataset.
- `--sample_row_num`: Number of rows in the input image.
- `--sample_col_num`: Number of columns in the input image.
- `--pad_size`: Padding size used during hologram generation.
- `--pixel_pitch`: Pixel pitch (distance between pixels) used in the hologram generation.
- `--wave_length`: Wavelength values for RGB channels.
- `--distance`: Distance used during holographic propagation.
- `--min_distance`: Minimum distance for propagation.
- `--max_distance`: Maximum distance for propagation.

### Larger Hologram Generation?

This model is able to generate holograms of different sizes for that it is a whole convolutional neural network. The size of the hologram is determined by the size of the input image.  

## Summary

With these commands, you can successfully train the model, generate phase-only holograms from RGBD images, and optionally propagate the holograms to create reconstructed images at various distances.
