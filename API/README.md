# Models Directory

This directory should contain your trained PyTorch model files.

## Required Files

- `best.pt` - The trained ASL recognition model
- `train_recognizer_torch.py` - The model class definition (should match the one used for training)

## Model Structure

The model expects:
- Input: Hand landmark features (21 landmarks × 3 coordinates = 63 features per frame)
- Output: Predicted ASL word/gesture with confidence score

## Getting the Model

If you don't have a trained model yet, you'll need to:
1. Collect ASL gesture data
2. Train the model using the training scripts
3. Place the resulting `best.pt` file in this directory

## Note

The `.gitignore` file is configured to exclude `.pt` files from version control due to their large size.
Consider using Git LFS or cloud storage for model file distribution.