import torch
import os

def inspect_checkpoint(filepath):
    print(f"Loaded: {filepath}")

    # Check file size and content type
    if filepath.endswith(".pt") and os.path.getsize(filepath) > 0:
        try:
            # First try regular torch.load
            ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
            print(f"Type: {type(ckpt)}")
            if isinstance(ckpt, dict):
                print("Keys:")
                for k in ckpt:
                    print(f"  {k}: {type(ckpt[k])}")
            else:
                print("Object attributes:")
                for attr in dir(ckpt):
                    if not attr.startswith("_"):
                        print(f"  {attr}")
        except RuntimeError as e:
            # Fallback to TorchScript
            print("Standard load failed, trying torch.jit.load() for TorchScript module...")
            try:
                scripted = torch.jit.load(filepath, map_location='cpu')
                print("Loaded TorchScript module.")
                print(f"Methods: {scripted.code}")
            except Exception as e2:
                print(f"Failed to load even with torch.jit.load: {e2}")
    else:
        print("File is empty or not a .pt file")

if __name__ == '__main__':
    inspect_checkpoint("model_36200.pt")
    print("#############")
    inspect_checkpoint("policy.pt")
