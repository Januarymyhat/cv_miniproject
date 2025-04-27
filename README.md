# Test Script for Single-Image Conditional GAN

This repository provides a test script to evaluate the trained conditional GAN model on the single-image generation task.

## Environment

Tested on the following environment:

- **Python**: 3.9.21
- **PyTorch**: 2.5.1
- **CUDA**: 12.1
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU


## Usage

1. Ensure that the trained model (`netG_final.pth`) is located in the `results/` directory.
2. Ensure that the input image (`jcsmr.jpg`) is placed in the `datasets/` directory.
3. Run the test script:


The script will:
- Load the trained generator model.
- Use a synthetic input image generated via TPS warp or edge map.
- Produce a generated image (`fake_test_output.png`) saved under `test_results/`.

## Output

After running, you will find:
- `test_results/input_test_image.png`: the structure-conditioned input
- `test_results/fake_test_output.png`: the generated output image

These images can be used for visual inspection or inserted into the project report for demonstration.

## Notes

- The model used here must be trained on the same augmentation strategy (e.g., TPS).
- If your model uses NeRF-style positional encoding, modify the `test.py` to include coordinate encoding in the input.

