# Evaluation Script

Simple object detection evaluation script that calculates mAP and accuracy metrics for COCO format datasets.

## What it does
- Calculates mAP@50, mAP@75, and mAP@50-95 metrics
- Shows class-wise accuracy 
- Generates confusion matrix
- Creates visualization plots
- Saves results to JSON file

## Requirements
```bash
pip install pycocotools numpy matplotlib seaborn
```

## Basic Usage
```bash
python evaluation.py ground_truth.json predictions.json
```

## Input Files
Both files should be in COCO JSON format with:
- `images` - list of image info
- `annotations` - list of bounding boxes and labels  
- `categories` - list of class names

## Output
- Console output with metrics
- `results.json` file with detailed results
- Plot windows showing confusion matrix and accuracy charts

## Example
```bash
python evaluation.py gt.json pred.json --results-file my_results.json
```

This will evaluate predictions against ground truth and save results to `my_results.json`.