#!/usr/bin/env python3
"""
Data Processing Tutorial: Annotation Matrix Generation and User Accuracy Analysis

This tutorial demonstrates how to process user annotation data from CSV files to:
1. Generate annotation matrices for collaborative labeling tasks
2. Calculate user accuracy scores and progression metrics
3. Handle bounding box overlap detection using IoU (Intersection over Union)
4. Process different types of annotation operations (new, edited, deleted)

Dataset Structure:
- Images: Contains metadata about images (id, dimensions, difficulty)
- User Annotations: Contains all user-submitted annotations with bounding boxes
- User Progression: Contains user performance metrics and accuracy scores
- User Image Assignments: Links users to their assigned images

Key Concepts Demonstrated:
- CSV data loading and processing with pandas
- Annotation matrix generation for collaborative labeling
- IoU calculation for bounding box overlap detection
- User performance analysis and scoring
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import json
import statistics
import sys
import logging
import os
from pathlib import Path
import argparse

def setup_configuration():
    """
    Setup all configuration parameters and create necessary directories.
    
    Returns:
        dict: A dictionary containing:
            - number_of_classes (int): Total number of classes
            - date (str): Current date in YYYY-MM-DD format
            - file_paths (dict): Dictionary of all file paths
            - categories (list): List of category dictionaries
            - logger (Logger): Configured logger instance
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data Processing Tutorial: Annotation Matrix Generation')
    parser.add_argument('--image-id', type=int, help='Specific image ID to process')
    parser.add_argument('--user-id', type=int, help='User ID to get accuracy score')
    parser.add_argument('--user-analysis', action='store_true', help='Get accuracy score for a specific user')
    parser.add_argument('--data-dir', type=str, default="./open_data",
                      help='Data directory path containing CSV files')
    parser.add_argument('--output-dir', type=str, default="./tutorial_output",
                      help='Output directory path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()

    # Number of annotation categories (vehicle types + background)
    number_of_classes = 15 + 1
    date = datetime.now().strftime("%Y-%m-%d")

    # Create output directories
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    results_dir = output_dir / "results"

    for dir_path in [output_dir, log_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # File paths for CSV data
    data_dir = Path(args.data_dir)
    file_paths = {
        'images_csv': str(data_dir / 'phase_2_image.csv'),
        'user_annotations_csv': str(data_dir / 'phase_2_user_annotation.csv'),
        'user_progression_csv': str(data_dir / 'phase_2_user_progression_score.csv'),
        'user_image_assignments_csv': str(data_dir / 'phase_2_user_image_user_annotation.csv'),
    }

    # Vehicle categories for annotation
    categories = [
        {"supercategory": "Vehicle", "id": i+1, "name": name} for i, name in enumerate([
            "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler", 
            "Two-wheeler", "LCV", "Mini-bus", "Mini-truck", "tempo-traveller", 
            "bicycle", "people", "white-swift-dzire"
        ])
    ]

    # Setup logging
    log_file = log_dir / f"tutorial_processing_{date}.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    return {
        'number_of_classes': number_of_classes,
        'date': date,
        'file_paths': file_paths,
        'categories': categories,
        'logger': logger,
        'args': args
    }

def load_csv_data(config):
    """
    Load CSV data files into pandas DataFrames.
    
    Args:
        config (dict): Configuration dictionary containing file paths and logger
        
    Returns:
        dict: Dictionary containing DataFrames for each CSV file
    """
    logger = config['logger']
    file_paths = config['file_paths']
    
    try:
        logger.info("Loading CSV data files...")
        
        # Load images data
        images_df = pd.read_csv(file_paths['images_csv'])
        logger.info(f"Loaded {len(images_df)} image records")
        
        # Load user annotations data
        annotations_df = pd.read_csv(file_paths['user_annotations_csv'])
        logger.info(f"Loaded {len(annotations_df)} annotation records")
        
        # Load user progression data
        progression_df = pd.read_csv(file_paths['user_progression_csv'])
        logger.info(f"Loaded {len(progression_df)} user progression records")
        
        # Load user image assignments
        assignments_df = pd.read_csv(file_paths['user_image_assignments_csv'])
        logger.info(f"Loaded {len(assignments_df)} user assignment records")
        
        return {
            'images_df': images_df,
            'annotations_df': annotations_df,
            'progression_df': progression_df,
            'assignments_df': assignments_df
        }
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    This is a key concept in computer vision for measuring overlap between bounding boxes.
    IoU = Area of Intersection / Area of Union
    
    Args:
        box1 (tuple): (x, y, width, height) of first box
        box2 (tuple): (x, y, width, height) of second box
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert from (x, y, width, height) to (x1, y1, x2, y2)
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def create_annotation_matrix(annotations_df, image_id, user_ids, logger):
    """
    Create annotation matrix for collaborative labeling analysis.
    
    This function demonstrates how to process collaborative annotation data where multiple
    users annotate the same image. The matrix shows which users annotated which bounding boxes
    and what categories they assigned.
    
    Matrix Structure:
    - Rows: User IDs
    - Columns: Bounding Box IDs
    - Values: Category IDs (or -1 if user didn't annotate that box)
    
    Args:
        annotations_df (DataFrame): User annotations data
        image_id (int): Image ID to filter by
        user_ids (list): List of user IDs to include
        logger: Logger instance
        
    Returns:
        tuple: (M, user_indices) where:
            - M (DataFrame): Annotation matrix with bbox_ids as columns and user_ids as rows
            - user_indices (dict): Mapping of user IDs to matrix indices
    """
    # Filter annotations for the specific image
    image_annotations = annotations_df[
        (annotations_df['image_id'] == image_id) & 
        (annotations_df['user_id'].isin(user_ids))
    ].copy()
    
    logger.info(f"Found {len(image_annotations)} annotations for image {image_id}")

    # Create user indices mapping
    user_indices = {user_id: i for i, user_id in enumerate(user_ids)}

    # Initialize dictionaries to store bbox information
    bbox_info = {}  # bbox_id -> (x, y, width, height)
    bbox_annotations = defaultdict(dict)  # bbox_id -> {user_id: category_id}

    # First pass: Process annotations with baseline_annotation_id (existing bboxes)
    baseline_annotations = image_annotations[image_annotations['baseline_annotation_id'].notna()]
    
    for _, record in baseline_annotations.iterrows():
        user_id = record['user_id']
        baseline_annotation_id = record['baseline_annotation_id']
        is_deleted = record['is_deleted']

        if is_deleted:
            # Mark as -1 if user deleted this bbox
            bbox_annotations[baseline_annotation_id][user_id] = -1
        else:
            # Store user_submitted_category_id and bbox info
            category_id = record['user_submitted_category_id']
            if pd.notna(category_id):
                bbox_annotations[baseline_annotation_id][user_id] = int(category_id)
                # Store bbox dimensions for IoU calculations
                x = record['x'] if pd.notna(record['x']) else 0
                y = record['y'] if pd.notna(record['y']) else 0
                width = record['width'] if pd.notna(record['width']) else 0
                height = record['height'] if pd.notna(record['height']) else 0
                bbox_info[baseline_annotation_id] = (x, y, width, height)

    # Second pass: Process new annotations (without baseline_annotation_id)
    new_annotations = image_annotations[
        (image_annotations['baseline_annotation_id'].isna()) & 
        (image_annotations['is_new'] == True) & 
        (image_annotations['is_deleted'] == False)
    ]
    
    for _, record in new_annotations.iterrows():
        user_id = record['user_id']
        
        # Get bbox dimensions
        x = record['x'] if pd.notna(record['x']) else 0
        y = record['y'] if pd.notna(record['y']) else 0
        width = record['width'] if pd.notna(record['width']) else 0
        height = record['height'] if pd.notna(record['height']) else 0
        category_id = record['user_submitted_category_id']
        record_id = record['id']

        # Skip if missing essential data
        if pd.isna(category_id) or pd.isna(record_id):
            logger.warning(f"Missing data in record {record_id}")
            continue

        new_box = (x, y, width, height)

        # Check for overlap with existing bboxes using IoU
        matched_bbox_id = None
        for bbox_id, bbox in bbox_info.items():
            iou = calculate_iou(new_box, bbox)
            if iou >= 0.8:  # 80% overlap threshold
                matched_bbox_id = bbox_id
                break

        if matched_bbox_id:
            # Use existing bbox_id if there's a match
            bbox_annotations[matched_bbox_id][user_id] = int(category_id)
        else:
            # Create a new bbox_id using record_id
            new_bbox_id = f"new_{int(record_id)}"
            bbox_annotations[new_bbox_id][user_id] = int(category_id)
            bbox_info[new_bbox_id] = new_box

            # Mark this bbox as not annotated (-1) for all other users
            for other_user_id in user_ids:
                if other_user_id != user_id and other_user_id not in bbox_annotations[new_bbox_id]:
                    bbox_annotations[new_bbox_id][other_user_id] = -1

    # Create DataFrame with bbox_ids as columns and user_ids as rows
    data = {}
    for bbox_id, user_annotations in bbox_annotations.items():
        column_data = []
        for user_id in user_ids:
            # Use -1 if user didn't annotate this bbox
            column_data.append(user_annotations.get(user_id, -1))
        data[f'bbox_id={bbox_id}'] = column_data

    # Create DataFrame with user_ids as indices (rows)
    if data:
        M = pd.DataFrame(data, index=user_ids)
    else:
        M = pd.DataFrame({f'bbox_id=empty': [-1] * len(user_ids)}, index=user_ids)
    
    logger.info(f"Generated annotation matrix with shape {M.shape}")
    return M, user_indices

def analyze_user_accuracy(user_id, image_id, data, logger):
    """
    Analyze user accuracy and performance metrics.
    
    This demonstrates how to extract user performance data and calculate
    accuracy scores based on progression metrics.
    
    Args:
        user_id (int): ID of the user
        image_id (int): ID of the image
        data (dict): Dictionary containing all DataFrames
        logger: Logger instance
        
    Returns:
        dict: Dictionary containing user performance metrics
    """
    # Get user assignment for this image
    assignment = data['assignments_df'][
        (data['assignments_df']['user_id'] == user_id) & 
        (data['assignments_df']['image_id'] == image_id)
    ]
    
    if assignment.empty:
        logger.error(f"No assignment found for user {user_id} and image {image_id}")
        return None
    
    assignment_record = assignment.iloc[0]
    zone_id = assignment_record['zone_id']
    level = assignment_record['level']
    
    # Get user progression data
    progression = data['progression_df'][
        (data['progression_df']['user_id'] == user_id) & 
        (data['progression_df']['zone_id'] == zone_id) & 
        (data['progression_df']['level'] == level)
    ]
    
    if progression.empty:
        logger.error(f"No progression data found for user {user_id}")
        return None
    
    progression_record = progression.iloc[0]
    
    return {
        'user_id': user_id,
        'level': level,
        'zone_id': zone_id,
        'ax_percentage_score': progression_record.get('ax_percentage_score', 0),
        'total_score_level': progression_record.get('total_score_level', 0),
        'sk_level': progression_record.get('sk_level', 0),
        'ax_k_level': progression_record.get('ax_k_level', 0)
    }

def process_image_annotations(image_id, data, logger):
    """
    Process and analyze annotations for a specific image.
    
    This function demonstrates the complete workflow of processing
    collaborative annotation data for a single image.
    
    Args:
        image_id (int): ID of the image to process
        data (dict): Dictionary containing all DataFrames
        logger: Logger instance
        
    Returns:
        tuple: (matrix, user_metrics) where matrix is the annotation matrix
               and user_metrics contains performance data for each user
    """
    # Get all users who annotated this image
    user_ids = sorted(data['annotations_df'][
        data['annotations_df']['image_id'] == image_id
    ]['user_id'].unique())
    
    if len(user_ids) == 0:
        logger.error(f"No users found for image {image_id}")
        return None, None

    logger.info(f"Found {len(user_ids)} users who annotated image {image_id}")
    
    # Create annotation matrix
    matrix, user_indices = create_annotation_matrix(
        data['annotations_df'], image_id, user_ids, logger
    )
    
    # Get image metadata
    image_info = data['images_df'][data['images_df']['id'] == image_id]
    if not image_info.empty:
        image_record = image_info.iloc[0]
        logger.info(f"Image {image_id}: {image_record['width']}x{image_record['height']}, "
                   f"difficulty: {image_record['difficulty']}, "
                   f"scheduled users: {image_record['num_scheduled_users']}")
    
    return matrix, user_ids

def run_tutorial_examples(config, data):
    """
    Run various tutorial examples to demonstrate the data processing capabilities.
    
    Args:
        config (dict): Configuration dictionary
        data (dict): Dictionary containing all DataFrames
    """
    logger = config['logger']
    
    print("\n" + "="*80)
    print("DATA PROCESSING TUTORIAL: ANNOTATION MATRIX GENERATION")
    print("="*80)
    
    # Example 1: Show dataset overview
    print(f"\n1. DATASET OVERVIEW")
    print(f"   - Images: {len(data['images_df']):,} records")
    print(f"   - User Annotations: {len(data['annotations_df']):,} records")
    print(f"   - User Progression: {len(data['progression_df']):,} records")
    print(f"   - User Assignments: {len(data['assignments_df']):,} records")
    
    # Example 2: Sample some images with annotations
    print(f"\n2. SAMPLE IMAGE ANALYSIS")
    sample_images = data['annotations_df']['image_id'].value_counts().sample(3)
    
    for image_id, annotation_count in sample_images.items():
        print(f"\n   Processing Image ID: {image_id} ({annotation_count} annotations)")
        
        matrix, user_ids = process_image_annotations(image_id, data, logger)
        if matrix is not None:
            print(f"   Generated matrix shape: {matrix.shape}")
            print(f"   Users involved: {user_ids}")
            
            # Show matrix preview
            print(f"   Matrix preview:")
            print(matrix.head())
            
            # Calculate agreement statistics
            if matrix.shape[1] > 0:  # If there are bounding boxes
                # Calculate how many users agreed on each bbox
                agreement_stats = []
                for col in matrix.columns:
                    non_missing = matrix[col][matrix[col] != -1]
                    if len(non_missing) > 1:
                        # Check if all non-missing values are the same
                        agreement = len(set(non_missing)) == 1
                        agreement_stats.append(agreement)
                
                if agreement_stats:
                    agreement_rate = sum(agreement_stats) / len(agreement_stats)
                    print(f"   Agreement rate: {agreement_rate:.2%}")
        
        print(f"   " + "-"*50)

def main():
    """
    Main function to run the data processing tutorial.
    """
    start_time = time.time()
    
    # Setup configuration
    config = setup_configuration()
    logger = config['logger']
    args = config['args']
    
    logger.info("Starting Data Processing Tutorial")
    
    try:
        # Load CSV data
        data = load_csv_data(config)
        if not data:
            logger.error("Failed to load CSV data")
            sys.exit(1)
        
        if args.image_id and args.user_analysis and args.user_id:
            # User accuracy analysis
            print(f"\n=== USER ACCURACY ANALYSIS ===")
            user_metrics = analyze_user_accuracy(args.user_id, args.image_id, data, logger)
            if user_metrics:
                print(f"User ID: {user_metrics['user_id']}")
                print(f"Level: {user_metrics['level']}")
                print(f"Zone ID: {user_metrics['zone_id']}")
                print(f"AX Percentage Score: {user_metrics['ax_percentage_score']:.10f}")
                print(f"Total Score Level: {user_metrics['total_score_level']}")
                
        elif args.image_id:
            # Single image analysis
            print(f"\n=== SINGLE IMAGE ANALYSIS ===")
            matrix, user_ids = process_image_annotations(args.image_id, data, logger)
            if matrix is not None:
                print(f"\nAnnotation Matrix for Image {args.image_id}:")
                print(matrix)
                
                print(f"\nMatrix as List of Lists:")
                matrix_list = matrix.values.tolist()
                for i, row in enumerate(matrix_list):
                    print(f"User {user_ids[i]}: {row}")
                    
        else:
            # Run full tutorial examples
            run_tutorial_examples(config, data)
        
        logger.info(f"Tutorial completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in tutorial: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 


