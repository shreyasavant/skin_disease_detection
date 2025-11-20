#!/usr/bin/env bash
set -euo pipefail

# Skin disease classifier: training + evaluation runner
# Usage: ./run_pipeline.sh [-t MODEL_TYPE] [-d DATASET_DIR] [-e EPOCHS]

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
MODEL_TYPE="cnn"
DATASET_DIR="$PROJECT_DIR/dataset"
IMAGE_SIZE=224
BATCH_SIZE=32
EPOCHS=10

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -t MODEL_TYPE    cnn or vit (default: cnn)
  -d DATASET_DIR   Dataset path (default: dataset/)
  -s IMAGE_SIZE    Image size (default: 224)
  -b BATCH_SIZE    Batch size (default: 32)
  -e EPOCHS        Epochs (default: 10)
  -h               Help

Examples:
  $0 -t cnn
  $0 -t vit -e 15
EOF
  exit 0
}

setup_dataset() {
  # Check if dataset already exists with class folders
  if [[ -d "$DATASET_DIR/train" && -d "$DATASET_DIR/test" ]]; then
    # Check if train and test directories have class subfolders
    train_classes=$(find "$DATASET_DIR/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    test_classes=$(find "$DATASET_DIR/test" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    
    if [[ $train_classes -gt 0 && $test_classes -gt 0 ]]; then
      echo "Using existing dataset at: $DATASET_DIR (found $train_classes classes in train/)"
      return 0
    else
      echo "Dataset directories exist but no class folders found. Downloading..."
    fi
  fi
  
  echo "Dataset not found - downloading..."
  local zip_file="$DATASET_DIR/skindiseasedataset.zip"
  local kaggle_url="https://www.kaggle.com/api/v1/datasets/download/pacificrm/skindiseasedataset"
  
  mkdir -p "$DATASET_DIR"
  
  # Try direct download
  if curl -fL -o "$zip_file" "$kaggle_url" 2>/dev/null && [[ -s "$zip_file" ]]; then
    echo "Download complete, extracting..."
    local tmp_dir="$DATASET_DIR/.tmp_$$"
    mkdir -p "$tmp_dir"
    unzip -q "$zip_file" -d "$tmp_dir"
    
    local train_dir=$(find "$tmp_dir" -type d -iname train -print -quit)
    local test_dir=$(find "$tmp_dir" -type d -iname test -print -quit)
    
    if [[ -n "$train_dir" && -n "$test_dir" ]]; then
      mkdir -p "$DATASET_DIR/train" "$DATASET_DIR/test"
      mv "$train_dir"/* "$DATASET_DIR/train/" 2>/dev/null || true
      mv "$test_dir"/* "$DATASET_DIR/test/" 2>/dev/null || true
      rm -rf "$tmp_dir" "$zip_file"
      echo "Dataset setup complete"
      return 0
    fi
    rm -rf "$tmp_dir"
  fi
  
  cat >&2 <<EOF

  Automatic download failed. Please download manually:

  1. Go to: https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
  2. Download the dataset
  3. Extract to: $DATASET_DIR
  4. Ensure train/ and test/ folders exist with class subfolders inside

  Or use Kaggle CLI:
    pip install kaggle
    kaggle datasets download -d pacificrm/skindiseasedataset
    unzip skindiseasedataset.zip -d $DATASET_DIR
EOF
  exit 1
}

while getopts "t:d:s:b:e:h" opt; do
  case $opt in
    t) MODEL_TYPE="$OPTARG" ;;
    d) DATASET_DIR="$OPTARG" ;;
    s) IMAGE_SIZE="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    h) usage ;;
    *) echo "Invalid option. Use -h for help" >&2; exit 1 ;;
  esac
done

[[ "$MODEL_TYPE" =~ ^(cnn|vit)$ ]] || { echo "MODEL_TYPE must be cnn or vit" >&2; exit 1; }

# Setup dataset
setup_dataset

mkdir -p "$PROJECT_DIR/models"

cd "$PROJECT_DIR/$MODEL_TYPE"
python3 "${MODEL_TYPE}.py" \
  --dataset_dir "$DATASET_DIR" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Training complete!"