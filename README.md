
```bash
git clone https://github.com/shreyasavant/skin_disease_detection
python -m venv venv
source venv/bin/activate  # For Linux/Mac
pip install -r requirements.txt
```

## Running the Pipeline

### Basic Usage

```bash
./run_pipeline.sh -t cnn          # Train CNN model with default settings
./run_pipeline.sh -t vit          # Train ViT model with default settings
```

### Custom Training Parameters

```bash
# Train CNN with more epochs
./run_pipeline.sh -t cnn -e 20

# Train ViT with custom epochs and batch size
./run_pipeline.sh -t vit -e 15 -b 16

# Train with larger image size for better resolution
./run_pipeline.sh -t cnn -s 256 -b 16

# Train with custom dataset path
./run_pipeline.sh -t cnn -d /path/to/dataset
```

### Advanced Recipes

```bash
# High-resolution training with smaller batch size
./run_pipeline.sh -t vit -s 384 -b 8 -e 25

# Fast training with smaller images and larger batches
./run_pipeline.sh -t cnn -s 128 -b 64 -e 5

# Full pipeline with all custom parameters
./run_pipeline.sh -t vit -d ./my_dataset -s 224 -b 32 -e 20
```

### Options Reference

- `-t MODEL_TYPE`: Model type (`cnn` or `vit`, default: `cnn`)
- `-d DATASET_DIR`: Dataset directory path (default: `./dataset`)
- `-s IMAGE_SIZE`: Input image size (default: `224`)
- `-b BATCH_SIZE`: Training batch size (default: `32`)
- `-e EPOCHS`: Number of training epochs (default: `10`)
- `-h`: Show help message

# project

### potential dataset
https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
link: https://drive.google.com/file/d/12rbs6TcvRGC3jw9pJt533BfKBNWQE-q2/view?usp=sharing
