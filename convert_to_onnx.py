import torch
import torch.onnx
from conditional_gan_pytorch import Generator
import argparse


def convert_to_onnx(checkpoint_path, onnx_path):
    # Load the trained model
    generator = Generator()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 512, 512)

    # Export the model
    torch.onnx.export(
        generator,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("onnx_path", type=str)
    args = parser.parse_args()

    convert_to_onnx(args.checkpoint_path, args.onnx_path)
