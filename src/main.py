import sys
import torch
from alphapack import PolicyValueNetwork


def main():
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} OUTPUT_MODEL_FILE", file=sys.stderr)
    sys.exit(1)

  output_path = sys.argv[1]

  model = PolicyValueNetwork().to("cuda")
  model.eval()
  model = torch.jit.script(model)
  model = torch.jit.freeze(model)
  model = torch.jit.optimize_for_inference(model)

  model.save(output_path)
  print(f"Saved TorchScript model to: {output_path}")

  print("Testing")
  model.to("cuda")
  image_input = torch.ones((1, 2, 10, 10), dtype=torch.float32, device="cuda")
  additional_input = torch.ones((1, 64), dtype=torch.float32, device="cuda")
  output = model(image_input, additional_input)
  policy_output_gpu, value_output_gpu = output
  print("policy_output_shape:", policy_output_gpu.shape)
  print("value_output_shape:", value_output_gpu.shape)


if __name__ == "__main__":
  main()
