import argparse
import torch
from PIL import Image
from torchvision import transforms
from conditional_gan_pytorch import Generator


def load_model(checkpoint_path):
    generator = Generator()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def postprocess_image(tensor):
    return transforms.ToPILImage()((tensor * 0.5 + 0.5).clamp(0, 1).squeeze(0))


def run_inference(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output


def main():
    parser = argparse.ArgumentParser("Load PyTorch .pth checkpoint and run inference")

    parser.add_argument("checkpoint_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("input_image_path", type=str, help="Path to input image")
    parser.add_argument("output_image_path", type=str, help="Path to save output image")
    args = parser.parse_args()

    # Load model
    generator = load_model(args.checkpoint_path)

    # Preprocess input image
    input_tensor = preprocess_image(args.input_image_path)

    # Print stats of the image (min/max values)
    print(f"Input min: {input_tensor.min()}")
    print(f"Input max: {input_tensor.max()}")
    print(f"Input mean: {input_tensor.mean()}")
    print(f"Input std: {input_tensor.std()}")

    # Run inference
    output_tensor = run_inference(generator, input_tensor)

    # Postprocess and save output image
    output_image = postprocess_image(output_tensor)
    output_image.save(args.output_image_path)
    print(f"Output saved to {args.output_image_path}")


if __name__ == "__main__":
    main()
