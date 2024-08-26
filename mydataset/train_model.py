import os
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator

from register_dataset import register_modis_dataset

def main():
    # Register the dataset
    register_modis_dataset()

    # Create config
    cfg = get_cfg()
    cfg.merge_from_file("C:/Users/sumed/detectron/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("modis_train",)
    cfg.DATASETS.TEST = ("modis_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "C:/Users/sumed/detectron/detectron2/weights/model_final_280758.pkl"  # Path to pretrained model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # Learning rate
    cfg.SOLVER.MAX_ITER = 1000  # Maximum number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Batch size per image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (Narendra Modi)
    cfg.MODEL.DEVICE = "cpu"  # Running on CPU for now

    # Ensure the model runs on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Training
    trainer.train()

if __name__ == "__main__":
    main()
