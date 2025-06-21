import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os
import argparse
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from contextlib import redirect_stdout
from io import StringIO
import time
from datetime import datetime
from pathlib import Path


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def make_coco_compatible(data):
    if 'info' not in data:
        data['info'] = {"description": "COCO dataset", "version": "1.0"}
    
    if 'licenses' not in data:
        data['licenses'] = []
    
    for ann in data.get('annotations', []):
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
        if 'area' not in ann and 'bbox' in ann:
            bbox = ann['bbox']
            ann['area'] = bbox[2] * bbox[3]
    
    return data

def align_image_ids(gt_data, pred_data):
    gt_filename_to_id = {img['file_name']: img['id'] for img in gt_data['images']}
    
    aligned_pred = copy.deepcopy(pred_data)
    id_mapping = {}
    valid_images = []
    
    for img in aligned_pred['images']:
        filename = img['file_name']
        if filename in gt_filename_to_id:
            old_id = img['id']
            new_id = gt_filename_to_id[filename]
            id_mapping[old_id] = new_id
            img['id'] = new_id
            valid_images.append(img)
    
    valid_annotations = []
    for ann in aligned_pred['annotations']:
        if ann['image_id'] in id_mapping:
            ann['image_id'] = id_mapping[ann['image_id']]
            valid_annotations.append(ann)
    
    aligned_pred['images'] = valid_images
    aligned_pred['annotations'] = valid_annotations
    
    return aligned_pred

def filter_classes(data, hide_classes):
    """Filter out specified classes from the dataset"""
    if not hide_classes:
        return data
    
    filtered_data = copy.deepcopy(data)
    hide_classes_set = set(hide_classes)
    
    # Filter annotations
    filtered_data['annotations'] = [
        ann for ann in filtered_data['annotations'] 
        if ann['category_id'] not in hide_classes_set
    ]
    
    # Filter categories
    filtered_data['categories'] = [
        cat for cat in filtered_data['categories'] 
        if cat['id'] not in hide_classes_set
    ]
    
    return filtered_data

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Find intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0

def get_class_accuracy_and_confusion_matrix(gt_data, pred_data, conf_thresh=0.25, iou_thresh=0.5):
    valid_preds = [p for p in pred_data['annotations'] if p.get('score', 0) > conf_thresh]
    
    gt_by_image = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    # Track matches for accuracy calculation
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    matched_preds = set()
    
    # Process each ground truth annotation
    for image_id, gt_anns in gt_by_image.items():
        for gt_ann in gt_anns:
            gt_class = gt_ann['category_id']
            gt_bbox = gt_ann['bbox']
            class_total[gt_class] += 1
            
            # Find best matching prediction
            best_iou = 0
            best_pred_class = None
            best_pred_idx = None
            
            for i, pred in enumerate(valid_preds):
                if pred['image_id'] == image_id and i not in matched_preds:
                    iou = calculate_iou(pred['bbox'], gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_class = pred['category_id']
                        best_pred_idx = i
            
            # Record result
            if best_iou > iou_thresh:
                if best_pred_class == gt_class:
                    class_correct[gt_class] += 1
                confusion_matrix[gt_class][best_pred_class] += 1
                matched_preds.add(best_pred_idx)
            else:
                confusion_matrix[gt_class]['miss'] += 1
    
    # Calculate accuracies
    class_accuracies = {}
    for class_id in class_total:
        accuracy = class_correct[class_id] / class_total[class_id]
        class_accuracies[class_id] = accuracy
    
    overall_accuracy = sum(class_correct.values()) / sum(class_total.values()) if sum(class_total.values()) > 0 else 0
    
    return class_accuracies, overall_accuracy, dict(confusion_matrix)


def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """Plot confusion matrix using matplotlib and seaborn"""
    
    # Get all unique classes (excluding 'miss')
    all_classes = set()
    for gt_class in confusion_matrix:
        all_classes.add(gt_class)
        for pred_class in confusion_matrix[gt_class]:
            if pred_class != 'miss':
                all_classes.add(pred_class)
    
    all_classes = sorted(list(all_classes))
    n_classes = len(all_classes)
    
    # Create confusion matrix array
    cm_array = np.zeros((n_classes + 1, n_classes + 1))  # +1 for 'miss' category
    
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    class_to_idx['miss'] = n_classes
    
    # Fill the confusion matrix
    for gt_class in confusion_matrix:
        if gt_class in class_to_idx:
            gt_idx = class_to_idx[gt_class]
            for pred_class, count in confusion_matrix[gt_class].items():
                if pred_class in class_to_idx:
                    pred_idx = class_to_idx[pred_class]
                    cm_array[gt_idx, pred_idx] = count
    
    # Create labels
    labels = [class_names.get(cls, f"Class_{cls}") for cls in all_classes] + ['Miss']
    
    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Use seaborn for better visualization
    sns.heatmap(cm_array, 
                annot=True, 
                fmt='g', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix\n(Rows: Ground Truth, Columns: Predictions)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm_array, labels


def plot_class_accuracies(class_accuracies, class_names, save_path=None):
    """Plot class-wise accuracies as a bar chart"""
    
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    class_labels = [class_names.get(cls, f"Class_{cls}") for cls in classes]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(classes)), accuracies, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Class-wise Detection Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), class_labels, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracies plot saved to: {save_path}")
    
    plt.show()


def plot_map_comparison(map_metrics, save_path=None):
    """Plot mAP comparison as a bar chart"""
    
    metric_names = ['mAP@50', 'mAP@75', 'mAP@50-95']
    metric_values = [map_metrics['map_50'], map_metrics['map_75'], map_metrics['map_50_95']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['lightcoral', 'lightblue', 'lightgreen'], alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.xlabel('mAP Metrics', fontsize=12)
    plt.ylabel('mAP Score', fontsize=12)
    plt.title('Mean Average Precision (mAP) Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, max(metric_values) * 1.2 if max(metric_values) > 0 else 0.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"mAP comparison plot saved to: {save_path}")
    
    plt.show()


def get_coco_map_all_metrics(gt_data, pred_data):
    """Calculate COCO mAP@50, mAP@75, and mAP@50:95 without printing detailed metrics"""
    
    # Convert predictions to COCO results format
    results = []
    for ann in pred_data['annotations']:
        results.append({
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'score': ann.get('score', 1.0)
        })
    
    # If no annotations left after filtering, return zero metrics
    if not results or not gt_data['annotations']:
        return {
            'map_50_95': 0.0,
            'map_50': 0.0,
            'map_75': 0.0
        }
    
    # Create temporary files for COCO API
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_temp:
        json.dump(gt_data, gt_temp)
        gt_path = gt_temp.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_temp:
        json.dump(results, pred_temp)
        pred_path = pred_temp.name
    
    try:
        # Run COCO evaluation with suppressed output
        coco_gt = COCO(gt_path)
        coco_pred = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        
        # Suppress all COCO evaluation output
        with redirect_stdout(StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        # Extract different mAP metrics
        map_50_95 = coco_eval.stats[0]  # mAP@0.50:0.95
        map_50 = coco_eval.stats[1]     # mAP@0.50
        map_75 = coco_eval.stats[2]     # mAP@0.75
        
        map_metrics = {
            'map_50_95': map_50_95,
            'map_50': map_50,
            'map_75': map_75
        }
        
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)
    
    return map_metrics


def save_results_to_json(results, output_file='results.json'):
    """Save evaluation results to JSON file in the specified format"""
    
    # Create the results dictionary in the requested format
    results_json = {
        "timestamp": results['timing']['timestamp'],
        "evaluation_time_seconds": results['timing']['evaluation_time'],
        "map_50_95": results['map_metrics']['map_50_95'],
        "map_50": results['map_metrics']['map_50'],
        "map_75": results['map_metrics']['map_75'],
        "overall_accuracy": results['overall_accuracy'],
        "class_acc": [
            {str(class_id): accuracy} 
            for class_id, accuracy in results['class_accuracies'].items()
        ],
        "categories": [
            {str(class_id): class_name} 
            for class_id, class_name in results['class_names'].items()
        ]
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return results_json


def evaluate_detection(gt_file, pred_file, conf_thresh=0.25, iou_thresh=0.5, 
                      show_plots=True, save_plots=False, save_results=True, 
                      results_file='results.json', hide_classes=None):
    """Main evaluation function"""
    
    # Start timing
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    gt_data = load_json(gt_file)
    pred_data = load_json(pred_file)
    
    # Prepare data
    gt_data = make_coco_compatible(gt_data)
    pred_data = make_coco_compatible(pred_data)
    pred_data = align_image_ids(gt_data, pred_data)
    
    # Get class names before filtering
    all_class_names = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Print information about hidden classes
    if hide_classes:
        hidden_class_names = [all_class_names.get(cls, f"Class_{cls}") for cls in hide_classes]
        print(f"Hiding classes from mAP calculation: {hide_classes} ({hidden_class_names})")
    
    # Filter out hidden classes for mAP calculation
    gt_data_filtered = filter_classes(gt_data, hide_classes)
    pred_data_filtered = filter_classes(pred_data, hide_classes)
    
    # Get filtered class names for mAP calculation
    class_names = {cat['id']: cat['name'] for cat in gt_data_filtered['categories']}
    
    # Calculate mAP metrics on filtered data
    map_metrics = get_coco_map_all_metrics(gt_data_filtered, pred_data_filtered)
    
    # Calculate class accuracies and confusion matrix on original data (not filtered)
    class_acc, overall_acc, confusion = get_class_accuracy_and_confusion_matrix(
        gt_data, pred_data, conf_thresh, iou_thresh
    )
    
    # End timing
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Print results
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Ground Truth File: {gt_file}")
    print(f"Predictions File: {pred_file}")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Timestamp: {timestamp}")
    print(f"Confidence Threshold: {conf_thresh}")
    print(f"IoU Threshold: {iou_thresh}")
    if hide_classes:
        print(f"Hidden Classes: {hide_classes} (excluded from mAP calculation)")
    print("-"*70)
    print("mAP METRICS (on filtered classes):")
    print(f"  mAP@0.50-0.95: {map_metrics['map_50_95']:.4f}")
    print(f"  mAP@0.50:      {map_metrics['map_50']:.4f}")
    print(f"  mAP@0.75:      {map_metrics['map_75']:.4f}")
    print("-"*70)
    print(f"Overall Classification Accuracy (all classes): {overall_acc:.4f}")
    
    print(f"\nClass-wise Accuracy (all classes):")
    print("-" * 40)
    for class_id, accuracy in class_acc.items():
        class_name = all_class_names.get(class_id, f"Class_{class_id}")
        hidden_indicator = " [HIDDEN FROM mAP]" if hide_classes and class_id in hide_classes else ""
        print(f"{class_name:<25}: {accuracy:.4f}{hidden_indicator}")
    
    # Prepare results dictionary
    results = {
        'map_metrics': map_metrics,
        'overall_accuracy': overall_acc,
        'class_accuracies': class_acc,
        'confusion_matrix': confusion,
        'class_names': all_class_names,  # Use all class names for results
        'hidden_classes': hide_classes or [],
        'timing': {
            'timestamp': timestamp,
            'evaluation_time': evaluation_time
        }
    }
    
    # Save results to JSON file
    if save_results:
        save_results_to_json(results, results_file)
    
    # Generate plot file names based on results file
    if save_plots:
        results_path = Path(results_file)
        base_name = results_path.stem
        plot_dir = results_path.parent
        
        cm_save_path = plot_dir / f"{base_name}_confusion_matrix.png"
        acc_save_path = plot_dir / f"{base_name}_class_accuracies.png"
        map_save_path = plot_dir / f"{base_name}_map_comparison.png"
    else:
        cm_save_path = acc_save_path = map_save_path = None
    
    # Plot visualizations (using all classes, not filtered)
    if show_plots or save_plots:
        if show_plots or save_plots:
            cm_array, cm_labels = plot_confusion_matrix(confusion, all_class_names, cm_save_path)
            plot_class_accuracies(class_acc, all_class_names, acc_save_path)
            plot_map_comparison(map_metrics, map_save_path)
    
    print(f"\nConfusion Matrix (Text) - All Classes:")
    print("-" * 40)
    for gt_class, predictions in confusion.items():
        gt_name = all_class_names.get(gt_class, f"Class_{gt_class}")
        hidden_indicator = " [HIDDEN FROM mAP]" if hide_classes and gt_class in hide_classes else ""
        print(f"{gt_name}{hidden_indicator}:")
        for pred_class, count in predictions.items():
            if pred_class == 'miss':
                print(f"  -> Missed: {count}")
            else:
                pred_name = all_class_names.get(pred_class, f"Class_{pred_class}")
                print(f"  -> {pred_name}: {count}")
        print()
    
    return results


def generate_output_filename(pred_file, base_output_dir=None):
    """Generate output filename based on prediction file name"""
    pred_path = Path(pred_file)
    base_name = pred_path.stem
    
    if base_output_dir:
        output_dir = Path(base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{base_name}_results.json"
    else:
        return pred_path.parent / f"{base_name}_results.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Object Detection Evaluation with Multiple mAP Metrics, Timing, and Multiple Files Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction file
  python test.py gt.json pred.json --results-file my_results.json
  
  # Multiple prediction files
  python test.py gt.json pred1.json pred2.json pred3.json
  
  # Multiple prediction files with custom output directory
  python test.py gt.json pred1.json pred2.json --output-dir results/
  
  # Hide classes from mAP calculation
  python test.py gt.json pred.json --hide-classes 1 3 5"""
    )
    
    # Required arguments
    parser.add_argument('ground_truth', 
                       help='Path to ground truth JSON file in COCO format')
    parser.add_argument('predictions', 
                       nargs='+',
                       help='Path(s) to prediction JSON file(s) in COCO format')
    
    # Optional arguments
    parser.add_argument('--results-file', '-r', 
                       help='Output file for results JSON (only used for single prediction file)')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for results (used for multiple prediction files)')
    parser.add_argument('--no-save-results', 
                       action='store_true',
                       help='Skip saving results to JSON file')
    parser.add_argument('--no-plots',
                       action='store_true',
                       help='Skip showing plots')
    parser.add_argument('--save-plots',
                       action='store_true',
                       help='Save plots as PNG files')
    parser.add_argument('--hide-classes', 
                       type=int, 
                       nargs='+',
                       help='Class IDs to hide from mAP calculation (space-separated list of integers)')
    parser.add_argument('--conf-thresh',
                       type=float,
                       default=0.25,
                       help='Confidence threshold for predictions (default: 0.25)')
    parser.add_argument('--iou-thresh',
                       type=float,
                       default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.predictions) > 1:
        if args.results_file:
            print("Warning: --results-file ignored when multiple prediction files are provided")
        if not args.output_dir:
            print("Using same directory as prediction files for output")
    
    return args


def validate_files(gt_file, pred_files):
    """Validate input files exist and are readable"""
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    # Try to load and validate ground truth JSON structure
    try:
        gt_data = load_json(gt_file)
        if 'annotations' not in gt_data or 'images' not in gt_data or 'categories' not in gt_data:
            raise ValueError(f"Ground truth file missing required COCO fields: {gt_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ground truth file {gt_file}: {e}")
    
    # Validate each prediction file
    for pred_file in pred_files:
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")
        
        try:
            pred_data = load_json(pred_file)
            if 'annotations' not in pred_data or 'images' not in pred_data:
                raise ValueError(f"Predictions file missing required fields: {pred_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in predictions file {pred_file}: {e}")


if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Validate input files
        validate_files(args.ground_truth, args.predictions)
        
        # Process each prediction file
        all_results = []
        total_start_time = time.time()
        
        print(f"Processing {len(args.predictions)} prediction file(s)...")
        print("="*70)
        
        for i, pred_file in enumerate(args.predictions, 1):
            print(f"\nProcessing file {i}/{len(args.predictions)}: {pred_file}")
            print("-"*50)
            
            # Determine output file
            if len(args.predictions) == 1 and args.results_file:
                output_file = args.results_file
            else:
                output_file = generate_output_filename(pred_file, args.output_dir)
            
            # Run evaluation
            results = evaluate_detection(
                gt_file=args.ground_truth,
                pred_file=pred_file,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
                show_plots=not args.no_plots,
                save_plots=args.save_plots,
                save_results=not args.no_save_results,
                results_file=str(output_file),
                hide_classes=args.hide_classes
            )
            
            # Store results for summary
            results['pred_file'] = pred_file
            results['output_file'] = str(output_file)
            all_results.append(results)
        
        # Print summary for multiple files
        if len(args.predictions) > 1:
            total_time = time.time() - total_start_time
            print(f"\n{'='*70}")
            print(f"SUMMARY FOR ALL {len(args.predictions)} FILES:")
            print(f"Total Processing Time: {total_time:.2f} seconds")
            print(f"{'='*70}")
            
            print(f"{'File':<30} {'mAP@50-95':<12} {'mAP@50':<12} {'mAP@75':<12} {'Accuracy':<10}")
            print("-"*70)
            
            for result in all_results:
                pred_name = Path(result['pred_file']).name
                print(f"{pred_name:<30} "
                      f"{result['map_metrics']['map_50_95']:<12.4f} "
                      f"{result['map_metrics']['map_50']:<12.4f} "
                      f"{result['map_metrics']['map_75']:<12.4f} "
                      f"{result['overall_accuracy']:<10.4f}")
            
            print(f"{'='*70}")
            print(f"Results saved to respective files in output directory")
        
        else:
            # Single file summary
            result = all_results[0]
            print(f"\n{'='*70}")
            print(f"SUMMARY:")
            if args.hide_classes:
                print(f"  Hidden Classes: {args.hide_classes}")
            print(f"  mAP@50-95: {result['map_metrics']['map_50_95']:.4f}")
            print(f"  mAP@50:    {result['map_metrics']['map_50']:.4f}")
            print(f"  mAP@75:    {result['map_metrics']['map_75']:.4f}")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.4f}")
            print(f"  Evaluation Time: {result['timing']['evaluation_time']:.2f} seconds")
            print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)