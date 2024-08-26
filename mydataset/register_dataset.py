import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog # used for registering and managing datasets
from detectron2.structures import BoxMode                   # for bounding box coordinates

def get_modis_dicts(img_dir, json_file):

    """
    Function which parses COCO (common objects in context) style JSON annotations and return Detectron2
    compatible dataset dictionaries

    COCO: A dataset format widely used in computer vision tasks

    Args:
    json_file: Path to COCO style json annotated files
    img_dir: Path to the images

    Returns:
    Dataset which is basically a list of dictionaries, each representing an image with its annotations    
    """

    with open(json_file) as f:
        imgs_anns = json.load(f)                            # load json content

    dataset_dicts = []                                      # initialize empty list of dictionaries
    for idx, v in enumerate(imgs_anns["images"]):           # enumerate helps keep track of index and element both
        record = {}                                         # initialize empty dictionary for image annontations
        
        # Construct key-value pairs for each image
        record["file_name"] = os.path.join(img_dir, v["file_name"])
        record["image_id"] = v["id"]
        record["height"] = v["height"]
        record["width"] = v["width"]
        # Fetching the corresponding annotation data for the image
        annos = [anno for anno in imgs_anns["annotations"] if anno["image_id"] == v["id"]] 
        objs = []
        # This will run only once as each image has a single annotation/bounding box but just for robustness
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],                    # bounding box coordinates (x,y,width,height)
                "bbox_mode": BoxMode.XYWH_ABS,           # specifying box mode (x,y,width,height) in absolute pixel values
                "category_id": anno["category_id"] - 1,  # detectron2 requires 0-indexed category id (Narendra Modi)
                "iscrowd": anno.get("iscrowd", 0)        # crowd of objects or not default to 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        # Now each element of the list will be a dictionary with both image specs and its annotation specs
    return dataset_dicts

def register_modis_dataset():
    img_dir = "C:/Users/sumed/detectron/detectron2/mydataset/images"  # The directory where images are stored
    json_file = "C:/Users/sumed/detectron/detectron2/mydataset/coco_annotations.json"
    
    # Check if the dataset is already registered and remove it if it is
    if "modis_train" in DatasetCatalog:
        DatasetCatalog.remove("modis_train")
    if "modis_val" in DatasetCatalog:
        DatasetCatalog.remove("modis_val")
    
    # Register the datasets
    DatasetCatalog.register("modis_train", lambda: get_modis_dicts(img_dir, json_file))
    DatasetCatalog.register("modis_val", lambda: get_modis_dicts(img_dir, json_file))
    
    # Set metadata for the datasets
    MetadataCatalog.get("modis_train").set(thing_classes=["Narendra Modi"])
    MetadataCatalog.get("modis_val").set(thing_classes=["Narendra Modi"])

register_modis_dataset()
