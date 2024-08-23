import argparse
from PIL import Image


def extract_images(source_image_path, output_image_path_prefix):
    img = Image.open(source_image_path)
    real_image = img.crop((0, 0, 512, 512))
    real_image_path = f"{output_image_path_prefix}_real.png"
    real_image.save(real_image_path)
    control_image_path = f"{output_image_path_prefix}_control.png"
    control_image = img.crop((512, 0, 1024, 512))
    control_image.save(control_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extract real and control images from a source image"
    )
    parser.add_argument("source_image_path", type=str, help="Path to source image")
    parser.add_argument(
        "output_image_path_prefix", type=str, help="Prefix for output image paths"
    )
    args = parser.parse_args()

    extract_images(args.source_image_path, args.output_image_path_prefix)
