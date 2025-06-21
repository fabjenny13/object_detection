# Scripts Directory

Contains data processing and evaluation scripts for the Urban Vision Hackathon.

## 1. Data Processing Script (`data_processing_tutorial.py`)

Processes collaborative annotation data from CSV files to generate annotation matrices and analyze user performance.

### What it does
- Creates annotation matrices from collaborative labeling data
- Calculates user accuracy and performance metrics  
- Analyzes annotation agreement between users
- Handles bounding box overlap detection using IoU
- Processes large CSV datasets efficiently

### Requirements
```bash
pip install pandas numpy
```

### Usage
```bash
# Analyze sample data and show overview
python data_processing_tutorial.py

# Analyze specific image annotations
python data_processing_tutorial.py --image-id 763187

# Get user performance metrics
python data_processing_tutorial.py --image-id 763187 --user-id 6483 --user-analysis

# Use custom data directory
python data_processing_tutorial.py --data-dir /path/to/csv/files

# Enable debug logging
python data_processing_tutorial.py --log-level DEBUG
```

### Input Files (CSV format)
- `phase_2_image.csv` - Image metadata (id, dimensions, difficulty)
- `phase_2_user_annotation.csv` - User annotations with bounding boxes
- `phase_2_user_progression_score.csv` - User performance tracking  
- `phase_2_user_image_user_annotation.csv` - User-image assignments

### Output
- Annotation matrices showing user agreements
- User accuracy scores and performance metrics
- Agreement statistics across collaborative annotations
- Log files with processing details

## 2. Evaluation Script (`evaluation.py`)

Simple object detection evaluation script that calculates mAP and accuracy metrics for COCO format datasets.

### What it does
- Calculates mAP@50, mAP@75, and mAP@50-95 metrics
- Shows class-wise accuracy 
- Generates confusion matrix
- Creates visualization plots
- Saves results to JSON file

### Requirements
```bash
pip install pycocotools numpy matplotlib seaborn
```

### Basic Usage
```bash
python evaluation.py ground_truth.json predictions.json
```

### Input Files
Both files should be in COCO JSON format with:
- `images` - list of image info
- `annotations` - list of bounding boxes and labels  
- `categories` - list of class names

### Output
- Console output with metrics
- `results.json` file with detailed results
- Plot windows showing confusion matrix and accuracy charts

### Example
```bash
python evaluation.py gt.json pred.json --results-file my_results.json
```

This will evaluate predictions against ground truth and save results to `my_results.json`.