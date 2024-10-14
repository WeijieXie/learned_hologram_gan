import argparse
import os

from learnedMethodForHologram.data_processor import read_exr_in_multi_folders

def process_folders(folders, channlesNum, height, width):
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder '{folder}' does not exist!")
        else:
            read_exr_in_multi_folders(folder, channlesNum, height, width)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EXR files in multiple folders.")

    parser.add_argument(
        'folders', metavar='F', type=str, nargs='+',
        help='The folders to process'
    )

    parser.add_argument('--channelsNum', type=int, default=None, help='Number of channels (e.g., 3)')
    parser.add_argument('--height', type=int, default=None, help='Height of the images (e.g., 192)')
    parser.add_argument('--width', type=int, default=None, help='Width of the images (e.g., 192)')

    args = parser.parse_args()

    if args.channelsNum is None:
        print("Error: channelsNum parameter is missing.")
        exit(1)
    if args.height is None:
        print("Error: height parameter is missing.")
        exit(1)
    if args.width is None:
        print("Error: width parameter is missing.")
        exit(1)

    process_folders(args.folders, args.channelsNum, args.height, args.width)
