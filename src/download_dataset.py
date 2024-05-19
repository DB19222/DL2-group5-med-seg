from datasets import load_dataset
import os

dataset = load_dataset("GoodBaiBai88/M3D-Seg", cache_dir=os.environ["TMPDIR"])

print(f"Dataset path: {os.environ['TMPDIR']}")
