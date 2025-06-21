import os
import json
import argparse

def create_coco_annotations(image_dir: str, output_dir: str) -> None:
    """
    Create and save COCO format annotations from object detection results.
    
    Args:
        image_dir: Directory containing the images
        output_dir: Directory to save the JSON output
    """
    # Initialize COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Define your categories here
    categories = [
        {"id": 1, "name": "Hatchback"},
        {"id": 2, "name": "Sedan"},
        {"id": 3, "name": "SUV"},
        {"id": 4, "name": "MUV"},
        {"id": 5, "name": "Bus"},
        {"id": 6, "name": "Truck"},
        {"id": 7, "name": "Three-wheeler"},
        {"id": 8, "name": "Two-wheeler"},
        {"id": 9, "name": "LCV"},
        {"id": 10, "name": "Mini-bus"},
        {"id": 11, "name": "Mini-truck"},
        {"id": 12, "name": "tempo-traveller"},
        {"id": 13, "name": "bicycle"},
        {"id": 14, "name": "Van"},
        {"id": 15, "name": "Others"}
    ]
    coco_format["categories"] = categories
    
    image_id = 0
    annotation_id = 0
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_file,
            "width": 1920,
            "height": 1080
        }
        coco_format["images"].append(image_info)
        
        # TODO: Perform object detection here
        # This is where you'll add your object detection code
        # The code should detect objects and create annotations
        
        # TODO: Create annotations for detected objects
        # Example annotation format:
        # annotation = {
        #     "id": annotation_id,
        #     "image_id": image_id,
        #     "category_id": category_id,
        #     "bbox": [x, y, width, height],
        #     "area": width * height,
        #     "segmentation": [],
        #     "iscrowd": 0
        # }
        # coco_format["annotations"].append(annotation)
        # annotation_id += 1
        
        image_id += 1
    
    # Save annotations to JSON file
    output_file = os.path.join(output_dir, 'output.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")


# NOTE: Please do not change the code below!
def main():
    parser = argparse.ArgumentParser(description='Create COCO format annotations from object detection results')
    parser.add_argument('--image_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save the COCO format JSON file')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and save COCO format annotations
    create_coco_annotations(args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()
