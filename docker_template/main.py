import os
import json
import argparse


from PIL import Image
from ultralytics import YOLO

def create_coco_annotations(image_dir: str, output_dir: str) -> None:
    """
    Create and save COCO format annotations from object detection results.
    
    Args:
        image_dir
        output_dir
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

        # -*- coding: utf-8 -*-


        import pandas as pd

        images_df = pd.read_csv("/home/hackathon/404_found/data/annotations/phase_2_image.csv")
        annotations_df = pd.read_csv("/home/hackathon/404_found/data/annotations/phase_2_user_annotation.csv")
        user_scores_df = pd.read_csv("/home/hackathon/404_found/data/annotations/phase_2_user_progression_score.csv")
        user_images_df = pd.read_csv("/home/hackathon/404_found/data/annotations/phase_2_user_image_user_annotation.csv")

        merged = images_df.merge(annotations_df, left_on = "id", right_on = "image_id")

        #keeping only high quality annotated data
        merged_users = user_scores_df.merge(user_images_df, on = "user_id")
        merged_users = merged_users(merged_users["ax_k_level"] > 0.7 )
        merged = merged.merge(merged_users, on = "image_id", how = "inner")

        #extracting bounding boxes

        import os

        from PIL import Image

        import shutil



        limit = int(0.7 * len(merged))  # 70% of the total rows

        for index, row in merged.iloc[:limit].iterrows():
            image_path = f"/home/hackathon/404_found/data/images/{row['image_name']}"


            try:
                img = Image.open(image_path)
                img_w, img_h = img.size
            except FileNotFoundError:
                #print(f"[Warning] Image not found: {image_path}. Skipping...")
                continue  # sk


            x = row['baseline_x']
            y = row['baseline_y']
            w = row['baseline_width']
            h = row['baseline_height']

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            annotation = f"{row['user_submitted_category_id'] - 1} {x_center} {y_center} {width} {height}\n"

            image_name_no_ext = os.path.splitext(row['image_name'])[0]

            with open(f"/home/hackathon/404_found/dataset/labels/train/{image_name_no_ext}.txt", "a") as f:
                f.write(annotation)


            source = f"/home/hackathon/404_found/data/{row['image_name']}.png"
            destination = f"/home/hackathon/404_found/datasets/images/train/{row['image_name']}.png"

            shutil.move(source, destination)



        for index, row in merged.iloc[limit:].iterrows():
            image_path = f"/home/hackathon/404_found/data/images/{row['image_name']}"


            try:
                img = Image.open(image_path)
                img_w, img_h = img.size
            except FileNotFoundError:
                #print(f"[Warning] Image not found: {image_path}. Skipping...")
                continue  # sk


            x = row['baseline_x']
            y = row['baseline_y']
            w = row['baseline_width']
            h = row['baseline_height']

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            annotation = f"{row['user_submitted_category_id'] - 1} {x_center} {y_center} {width} {height}\n"

            image_name_no_ext = os.path.splitext(row['image_name'])[0]

            with open(f"/home/hackathon/404_found/dataset/labels/val/{image_name_no_ext}.txt", "a") as f:
                f.write(annotation)

            source = f"/home/hackathon/404_found/data/{row['image_name']}.png"
            destination = f"/home/hackathon/404_found/datasets/images/val/{row['image_name']}.png"

            shutil.move(source, destination)

        """TRAIN MODEL"""

        from ultralytics import YOLO

        model = YOLO("yolo11s.pt")

        model.train(data="/home/hackathon/404_found/dataset/data.yaml", epochs=50, imgsz=640, save_period = 1)
        #metrics = model.val()
        
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


        model = YOLO("best.pt")  # path to your downloaded model

        results = model("/home/hackathon/404_found/data/val-images.jpg", save = True)  # or a folder of images
        results[0].show()  # to visualize

        # Replace this with your actual class list

        for i, result in enumerate(results):

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = xyxy
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                })

                annotation_id += 1

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
