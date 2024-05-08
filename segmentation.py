import os
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image



class PromptSAM(object):
    def __init__(self, checkpoint_name="FastSAM.pt", device=None):
        self.module_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.device = self.initialize_device(device)
        self.model = self.instantiate_model(self.module_dir, self.checkpoint_name)

    def segment(self):
        pass
    
    def annotate_inference(self, inference, area_threshold=0):
        """Returns list of annotation dicts"""
        annotations = []
        for i in range(len(inference.masks.data)):
            annotation = {}
            annotation["id"] = i
            annotation["segmentation"] = inference.masks.data[i].cpu().numpy()==1
            annotation["bbox"] = inference.boxes.data[i]
            annotation["score"] = inference.boxes.conf[i]
            annotation["area"] = annotation["segmentation"].sum()

            if annotation["area"] >= area_threshold:
                annotations += [annotation]
        return annotations
    
    def aggregate_masks(self, annotations, point_prompts, label_prompts):
        pass
        
    def initialize_device(self, device):
        """Initializes the device for inference
        Args:
            device (str): Device name 
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def instantiate_model(self, root, checkpoint_name):
        """Instantiates YOLO model based on checkpoint name
        Args:
            root (str): Directory where the checkpoint located
            checkpoint_name (str): Name of the checkpoint of YOLO
        """
        return YOLO(os.path.join(root, checkpoint_name))
    
    

        
    
