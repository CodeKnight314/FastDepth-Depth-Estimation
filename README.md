# FastDepth-Depth-Estimation

## Overview
This is an educational repository on the replication of FastDepth MobileV2 Encoder model for Monocular Depth estimation. The implementation is done primarily as an implementation exercise along with minor modifications of my own for exploratory purposes.

## Table of Contents
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Training](#training)
  - [Dataset Preparation](#dataset-preparation)

## Results

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/FastDepth-Depth-Estimation.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv fastdepth-env
    source fastdepth-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd FastDepth-Depth-Estimation/
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Inference
Use the `inference.py` script to perform depth estimation on RGB images in a specified directory.

**Arguments:**
- `--root`: Directory containing input images.
- `--output`: Directory to save super-resolved images.
- `--path`: Path to FastDepth model weights

**Example:**
```bash
python inference.py --root_dir ./data/input --output_dir ./data/output --path ./path/to/weights
```

### Evaluation
Use the `evaluate.py` script to perform evaluation on NYU test dataset with samples generation in a specified directory. 

**Arguments:**
- `--root`: Directory to preprocessed NYU Depth dataset with both train and test subsets.
- `--output`: Directory to store generated samples from depth estimation
- `--path`: Path to FastDepth model weights

**Example:**
```bash
python inference.py --root ./data/input --output ./data/output --path ./path/to/weights
```

### Training
Use the `train.py` script to train FastDepth on NYU Depth dataset with optional configs for learning rate and batch size.

**Arguments:**
- `--root`: Directory containing NYU Depth dataset
- `--output`: Directory for storing logs and saved weights
- `--epochs`: Number of epochs for training the model, default set to 20 epochs.
- `--lr`: Learning rate for model, default set to 1e-2 then decreased to 20% every 5 epochs
- `--batch`: Number of batches for model training, default set to 8.

**Example:**
```bash
python train.py --root ./data/input --output ./data/output
```

### Dataset Preparation
Use the `preprocess_nyu.py` to preprocess downloaded NYU Depth dataset and separate correctly Depth and RGB images.

**Arguments:**
- `--source`: Directory containing unzipped folder from NYU Depth dataset
- `--destination`: Directory to store train and test subset from NYU dataset, containing subdirectories of DEPTH and RGB.

**Example:**
```bash
python preprocess_nyu.py --source ./data/input --destination ./data/output
```