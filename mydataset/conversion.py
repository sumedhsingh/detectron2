import os
import json
import xml.etree.ElementTree as ET

def convert_voc_to_coco(voc_dir, output_file):
    # Define COCO dataset structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Narendra Modi"}]
    }
    annotation_id = 1

    for idx, filename in enumerate(os.listdir(voc_dir)):
        if not filename.endswith('.xml'):
            continue

        # Parse XML file
        tree = ET.parse(os.path.join(voc_dir, filename))
        root = tree.getroot()

        # Get image file name
        image_file = root.find('filename').text
        image_path = os.path.join(voc_dir.replace('annotations', 'images'), image_file)
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Add image info to COCO dataset
        image_info = {
            "id": idx + 1,
            "file_name": image_file,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)

        # Add annotation info to COCO dataset
        for obj in root.findall('object'):
            category = obj.find('name').text
            if category == "Narendra Modi":
                bndbox = obj.find('bndbox')
                bbox = [
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                    int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
                ]
                annotation = {
                    "id": annotation_id,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

    # Write COCO JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

if __name__ == "__main__":
    voc_dir = 'C:/Users/sumed/detectron/detectron2/mydataset/annotations'
    output_file = 'C:/Users/sumed/detectron/detectron2/mydataset/coco_annotations.json'
    convert_voc_to_coco(voc_dir, output_file)
