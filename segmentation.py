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
        self.create_dirs(self.module_dir)

    def segment(self, image_name, point_prompts, label_prompts, image_size):
        image = self.resize_image(self.read_image(os.path.join(self.module_dir, "segmentation-images", image_name)), image_size)
        print(image.shape)
        self.plot_prompt_points_on_image(image, point_prompts, label_prompts)
        aggregated_mask = self.aggregate_masks(self.annotate_inference(self.model(image[None], 
                                                                                  device=self.device, 
                                                                                  retina_masks=True)[0]), 
                                               point_prompts,
                                               label_prompts)
        self.paste_mask_on_image(image, aggregated_mask)
        
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
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), lambda x: x[:3]])(image)
    
    def plot_prompt_points_on_image(self, image, points, labels=None, fpath=None):
        """Plots prompt points onto given image and saves"""
        if labels is None:
            labels = [1]*len(points)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0).numpy())
        self._scatter_labeled_points(points, labels, ax)
        ax.axis('on')
        if fpath is None:
            fpath = os.path.join(self.module_dir, "segmented-images", "prompts_on_image.png")
        fig.savefig(fpath)
        
    def _scatter_labeled_points(self, points, labels, ax, marker_size=375):
        """Plots labeled points into ax object
        Args:
            points (list): List of 2D points in h, w form
            labels (list): List of labels
            ax: Axes object of pyplot
            marker_size (int): Size of the marker
        """
        points = torch.tensor(points)
        labels = torch.tensor(labels)
        pos_points = points[labels==1]
        neg_points = points[labels==0]
        ax.scatter(pos_points[:, 1], pos_points[:, 0], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 1], neg_points[:, 0], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
        
    def paste_mask_on_image(self, image, mask, alpha=0.7, fpath=None):
        """Adds masks on given image and saves
        Args:
            image (torch.Tensor): Image that be masked
            mask (torch.Tensor): Mask that be applied to image
            alpha (float): Alpha value for transparency. Default: 0.7
            fpath (str): Path of the image to which be saved
        """
        image = torch.cat([image, torch.ones((1, *image.shape[1:]))])
        image_mask = torch.zeros_like(image)
        image_mask[:,mask] = torch.cat([torch.rand(3), torch.Tensor([alpha])])[:, None]
        
        if fpath is None:
            fpath = os.path.join(self.module_dir, "segmented-images", "masked_image.png")
        Image.alpha_composite(transforms.functional.to_pil_image(image),
                              transforms.functional.to_pil_image(image_mask)).save(fpath)

    def create_dirs(self, root):
        """Creates directories used in segmentation
        Args:
            root (str): Root directory under which sub-directories be created
        """
        dir_names = ["segmented-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)





if __name__ == "__main__":
    prompt_sam = PromptSAM()
    point_prompts = [[500, 400], [500, 700], [700, 400]]
    point_labels = [0 ,0, 1]
    prompt_sam.segment("dogs.jpg", point_prompts, point_labels)