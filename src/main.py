import sys
import torch
from alphapack import PolicyValueNetwork


def main():
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} OUTPUT_MODEL_FILE", file=sys.stderr)
    sys.exit(1)

  output_path = sys.argv[1]

  model = PolicyValueNetwork().to("cuda")
  params_count = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {params_count}")

  model.eval()
  model = torch.jit.script(model)
  model = torch.jit.freeze(model)
  model = torch.jit.optimize_for_inference(model)

  model.save(output_path)
  print(f"Saved TorchScript model to: {output_path}")


if __name__ == "__main__":
  main()
