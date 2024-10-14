import argparse
import torch
from learnedMethodForHologram import utilities
from learnedMethodForHologram.watermelon_hologram.data_loader import (
    dataloaderImgDepth as data_loader,
)
from learnedMethodForHologram.watermelon_hologram.generator import Generator
from learnedMethodForHologram.angular_spectrum_method import (
    bandLimitedAngularSpectrumMethod_for_multiple_distances as BLASM_v4,
)


def main(args):
    # Create the data loader
    dataset_test = data_loader(
        img_path=args.img_path,
        depth_path=args.depth_path,
        samplesNum=args.samplesNum,  # Assumed dataset size, adjust as necessary
        channlesNum=3,
        height=args.sample_row_num,
        width=args.sample_col_num,
        cuda=True,
    )

    # Initialize the model
    model = Generator(
        sample_row_num=args.sample_row_num,
        sample_col_num=args.sample_col_num,
        pad_size=args.pad_size,
        filter_radius_coefficient=0.45,
        pixel_pitch=args.pixel_pitch,
        wave_length=torch.tensor(args.wave_length),
        distance=torch.tensor([args.distance]),
        pretrained_model_path=args.model_path,
        pretrained_model_path_RGBD2AP=None,
        pretrained_model_path_AP2POH=None,
    )
    model.eval()

    # Get input data (based on user-specified index)
    with torch.no_grad():
        img_depth = dataset_test[args.index].unsqueeze(0)
        POH = model(img_depth)

    # Save the generated POH data
    torch.save(POH.squeeze(0), args.poh_output_path)
    print(f"POH data saved at {args.poh_output_path}")

    # If propagation flag is set to True, perform propagation
    if args.propagate:
        propagator_test = BLASM_v4(
            sample_row_num=args.sample_row_num,
            sample_col_num=args.sample_col_num,
            pad_size=args.pad_size,
            distances=torch.linspace(
                args.min_distance, args.max_distance, args.num_intervals
            ),
            filter_radius_coefficient=args.filter_radius_coefficient,
            pixel_pitch=args.pixel_pitch,
            wave_length=torch.tensor(args.wave_length),
            band_limit=False,
            cuda=True,
        )

        amp_ones = torch.ones_like(POH).to("cuda")
        amp_hat = propagator_test(
            amp_ones,
            POH,
            torch.linspace(args.min_distance, args.max_distance, args.num_intervals),
        )

        # Use utilities.multi_sample_plotter to save propagated images
        utilities.multi_sample_plotter(
            utilities.tensor_normalizor_2D(amp_hat),
            titles=None,
            rgb_img=True,
            save_dir=args.output_image_dir,
        )
        print(f"Propagated images saved at {args.output_image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for generating and propagating POH"
    )

    # Required user input arguments
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the input img.bin file"
    )
    parser.add_argument(
        "--depth_path", type=str, required=True, help="Path to the input depth.bin file"
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the sample to generate POH for",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--poh_output_path",
        type=str,
        required=True,
        help="Path to save the generated POH",
    )

    # Optional generator parameters
    parser.add_argument("--samplesNum", type=int, default=100, help="Number of samples")
    parser.add_argument(
        "--sample_row_num", type=int, default=384, help="Number of sample rows"
    )
    parser.add_argument(
        "--sample_col_num", type=int, default=384, help="Number of sample columns"
    )
    parser.add_argument("--pad_size", type=int, default=320, help="Padding size")
    parser.add_argument(
        "--pixel_pitch", type=float, default=3.74e-6, help="Pixel pitch"
    )
    parser.add_argument(
        "--wave_length",
        nargs="+",
        type=float,
        default=[638e-9, 520e-9, 450e-9],
        help="Wavelengths for RGB channels",
    )
    parser.add_argument(
        "--distance", type=float, default=1e-3, help="Distance for propagation"
    )
    parser.add_argument(
        "--filter_radius_coefficient",
        type=float,
        default=0.35,
        help="Filter radius coefficient",
    )

    # Optional propagation parameters
    parser.add_argument(
        "--propagate", action="store_true", help="Flag to enable propagation"
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=4e-4,
        help="Minimum distance for propagation",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=10e-4,
        help="Maximum distance for propagation",
    )
    parser.add_argument(
        "--num_intervals",
        type=int,
        default=1,
        help="Number of intervals for propagation distances",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        default=None,
        help="Directory to save propagated images",
    )

    args = parser.parse_args()
    main(args)
