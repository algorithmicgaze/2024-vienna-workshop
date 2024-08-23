import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse


def load_onnx_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path)


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).numpy()


def postprocess_image(output):
    # Convert from NCHW to HWC format
    output = np.transpose(output, (1, 2, 0))
    # Denormalize
    output = (output * 0.5 + 0.5).clip(0, 1)
    # Convert to uint8
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output)


def run_inference(session, input_array):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_array})[0]


def main():
    parser = argparse.ArgumentParser("Load ONNX model and run inference")
    parser.add_argument("onnx_path", type=str, help="Path to ONNX model")
    parser.add_argument("input_image_path", type=str, help="Path to input image")
    parser.add_argument("output_image_path", type=str, help="Path to save output image")
    args = parser.parse_args()

    onnx_session = load_onnx_model(args.onnx_path)

    # Preprocess input image
    input_array = preprocess_image(args.input_image_path)
    # Add batch dimension
    input_array = np.expand_dims(input_array, axis=0)

    # Print stats of the image (min/max values)
    print(f"Input shape: {input_array.shape}")
    print(f"Input min: {input_array.min()}")
    print(f"Input max: {input_array.max()}")
    print(f"Input mean: {input_array.mean()}")
    print(f"Input std: {input_array.std()}")

    # Run inference
    output_array = run_inference(onnx_session, input_array)

    # Print stats of the output image (min/max values)
    print(f"Output shape: {output_array.shape}")
    print(f"Output min: {output_array.min()}")
    print(f"Output max: {output_array.max()}")
    print(f"Output mean: {output_array.mean()}")
    print(f"Output std: {output_array.std()}")

    # Postprocess and save output image
    output_image = postprocess_image(output_array[0])  # Remove batch dimension
    output_image.save(args.output_image_path)
    print(f"ONNX output saved to {args.output_image_path}")


if __name__ == "__main__":
    main()
