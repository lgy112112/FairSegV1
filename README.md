# FairSegV1

FairSegV1 is a fairness-focused image segmentation framework based on SAM (Segment Anything Model), designed to reduce bias in model predictions. By incorporating sensitive attribute prediction and fairness constraints, the project enhances model fairness while maintaining segmentation performance.

## Key Features

- Built on SAM model, supporting various ViT architectures
- Extensible fairness constraint module
- Supports distributed training
- Provides complete training and evaluation pipelines
- Visualization tools included

## Model Architecture

The core of FairSegV1 is the FairMedSAM model, which consists of the following components:

1. **Image Encoder**: Based on ViT architecture, extracts image features
2. **Prompt Encoder**: Processes input prompts (e.g., bounding boxes)
3. **Mask Decoder**: Generates segmentation masks
4. **Sensitive Attribute Predictor**: Predicts sensitive attributes for fairness constraints

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/FairSegV1.git
   cd FairSegV1
   ```

2. Install dependencies:
   ```bash
   python setup.py -e .
   ```


## Usage Instructions

### Training

1. Prepare the dataset, ensuring the directory structure is as follows:
   ```
   HarvardFairSeg/
     ├── Training/
     └── Test/
   ```

2. Modify parameters in the training script `train.sh` or `train_fair.sh`:
   - `cuda_devices`: Specify GPUs to use
   - `att_name`: Sensitive attribute name
   - `num_att`: Number of sensitive attribute categories
   - `exp_name`: Experiment name
   - `model_type`: `vit_b`, `vpt_vit_b`

3. Start training:
   ```bash
   bash train.sh
   ```

### Evaluation

After training, the model will automatically evaluate on the test set, with results saved in the `logs/` directory.

## Parameter Description

| Parameter | Description |
|-----------|-------------|
| `--tr_npy_path` | Path to training data |
| `--val_npy_path` | Path to validation data |
| `--task_name` | Task name |
| `--model_type` | Model type (vit_b/vpt_vit_b) |
| `--checkpoint` | Path to pre-trained model |
| `--attribute_name` | Sensitive attribute name |
| `--num_sensitive_classes` | Number of sensitive attribute categories |
| `--num_epochs` | Number of training epochs |
| `--batch_size` | Batch size |
| `--alpha` | Weight for sensitive attribute prediction loss |
| `--beta` | Weight for entropy loss |

## Result Visualization

Loss curves and evaluation metrics during training are automatically saved in the `work_dir/` directory.
