#!/usr/bin/env python3
"""
Memory-safe training wrapper that clears GPU memory before starting
"""
import os
import gc
import torch

# Set memory optimization env vars BEFORE importing anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

print("🧹 Clearing GPU memory...")

# Clear any existing CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    # Print GPU status
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"📊 GPU Memory - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")

print("✅ Memory cleared, starting training...\n")

# Now import and run the main training
from class_5_hw import main
import sys

# Parse args
quick_mode = '--quick' in sys.argv or '-q' in sys.argv

# Get methods
methods = None
if '--methods' in sys.argv:
    idx = sys.argv.index('--methods')
    if idx + 1 < len(sys.argv):
        methods = sys.argv[idx + 1].split(',')

# Get samples
num_samples = None
if '--samples' in sys.argv:
    idx = sys.argv.index('--samples')
    if idx + 1 < len(sys.argv):
        samples_arg = sys.argv[idx + 1]
        if samples_arg.lower() not in ['all', '0']:
            num_samples = int(samples_arg)

# Run training
main(quick_mode=quick_mode, num_samples=num_samples, methods=methods)
