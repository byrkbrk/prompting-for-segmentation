import os
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np



class PromptSAM(object):
    def __init__(self, checkpoint_name="FastSAM.pt", device=None):
        self.module_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.device = self.initialize_device(device)
        self.model = self.instantiate_model(self.module_dir, self.checkpoint_name)

    def segment(self, image_path, point_prompts, label_prompts):
        image = self.resize_image(self.read_image(image_path), 1024)
        self.plot_prompt_points_on_image(image, point_prompts, label_prompts)
        
    
    def annotate_inference(self, inference, area_threshold=0):
        """Returns list of annotation dicts
        Args:
            inference: Ouput of the model
            area_threshold (int): Threshold for the segmentation area
        """
        annotations = []
        for i in range(len(inference.masks.data)):
            annotation = {}
            annotation["id"] = i
            annotation["segmentation"] = inference.masks.data[i].cpu()==1
            annotation["bbox"] = inference.boxes.data[i]
            annotation["score"] = inference.boxes.conf[i]
            annotation["area"] = annotation["segmentation"].sum()

            if annotation["area"] >= area_threshold:
                annotations += [annotation]
        return annotations
    
    def aggregate_masks(self, annotations, point_prompts, label_prompts):
        filtered_mask = torch.zeros(annotations[0]["segmentation"].shape)
        for annotation in sorted(annotations, key=lambda x: x["area"], reverse=True):
            for point_prompt, label_prompt in zip(point_prompts, label_prompts):
                if annotation["segmentation"][*point_prompt]:
                    filtered_mask[annotation["segmentation"]] = 1 if label_prompt else 0
        return filtered_mask == 1
        
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
            root (str): Directory where the checkpoint is located
            checkpoint_name (str): Name of the checkpoint of YOLO
        """
        return YOLO(os.path.join(root, checkpoint_name))
    
    def read_image(self, img_path):
        """Returns Image object for given image path"""
        return Image.open(img_path)
    
    def resize_image(self, image, size):
        """Returns resized image"""
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])(image)
    
    def plot_prompt_points_on_image(self, image, points, labels=None, fpath=None):
        """Plots prompt points onto given image and saves"""
        if labels is None:
            labels = [1]*len(points)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0).numpy())
        self._scatter_labeled_points(points, labels, ax)
        ax.axis('on')
        if fpath is None:
            fpath = os.path.join(self.module_dir, "prompts_on_image.png")
        fig.savefig(fpath)
        
    def _scatter_labeled_points(self, points, labels, ax, marker_size=375):
        """Plots labeled points into ax object"""
        points = torch.tensor(points)
        labels = torch.tensor(labels)
        pos_points = points[labels==1]
        neg_points = points[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)

if __name__ == "__main__":
    prompt_sam = PromptSAM()
    point_prompts = [[500, 500], [1000, 800]]
    point_labels = [1, 0]
    prompt_sam.segment("dogs.jpg", point_prompts, point_labels)