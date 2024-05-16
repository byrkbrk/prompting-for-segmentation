import os
import torch
from torchvision import transforms
from ultralytics import FastSAM
from PIL import Image
from matplotlib import pyplot as plt



class PromptSAM(object):
    def __init__(self, image_name, checkpoint_name="FastSAM-x.pt", device=None):
        self.image_name = image_name
        self.module_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.device = self.initialize_device(device)
        self.model = self.instantiate_model(self.module_dir, self.checkpoint_name)
        self.create_dirs(self.module_dir)

    def segment(self, point_prompts, label_prompts, image_size):
        """Plots prompts and pastes masks onto given image and saves"""
        image = self.resize_image(self.read_image(os.path.join(self.module_dir, "segmentation-images", self.image_name)), image_size)
        annotations = self.annotate_inference(self.model(image[None], device=self.device, retina_masks=True)[0])
        self.plot_bbox_prompts_on_image(image, point_prompts, label_prompts)
        self.paste_mask_on_image(image, self.aggregate_bbox_masks(annotations, point_prompts, label_prompts), save=True)
        #self.plot_point_prompts_on_image(image, point_prompts, label_prompts)
        #self.paste_mask_on_image(image, self.aggregate_masks(annotations, point_prompts, label_prompts), save=True)
        self.paste_multiple_masks_on_image(image, annotations)
        
    def annotate_inference(self, inference, area_threshold=0):
        """Returns list of annotation dicts
        Args:
            inference: Output of the model
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
        """Aggregates masks based on given prompts"""
        if label_prompts is None: label_prompts = [1]*len(point_prompts)

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
        return FastSAM(os.path.join(root, checkpoint_name))
    
    def read_image(self, img_path):
        """Returns Image object for given image path"""
        return Image.open(img_path)
    
    def resize_image(self, image, size):
        """Returns resized image"""
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), lambda x: x[:3]])(image)
    
    def plot_point_prompts_on_image(self, image, points, labels=None, fpath=None):
        """Plots prompt points onto given image and saves"""
        if labels is None:
            labels = [1]*len(points)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0).numpy())
        self._scatter_labeled_points(points, labels, ax)
        ax.axis('on')
        if fpath is None:
            fpath = os.path.join(self.module_dir, "segmented-images", 
                                 os.path.splitext(self.image_name)[0] + "_point_prompts_on_image.png")
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
        
    def paste_mask_on_image(self, image, mask, alpha=0.6, fpath=None, save=False):
        """Adds masks on given image and returns 4-channel tensor
        Args:
            image (torch.Tensor): Image that be masked
            mask (torch.Tensor): Mask that be applied to image
            alpha (float): Alpha value for transparency. Default: 0.6
            fpath (str): Path of the image to which be saved
            save (bool): Option to save the resultant image. Default: False
        """
        if image.shape[0] == 3:
            image = torch.cat([image, torch.ones((1, *image.shape[1:]))])
        image_mask = torch.zeros_like(image)
        image_mask[:,mask] = torch.cat([torch.rand(3), torch.Tensor([alpha])])[:, None]
        image = Image.alpha_composite(transforms.functional.to_pil_image(image),
                                      transforms.functional.to_pil_image(image_mask))
        
        if save:
            if fpath is None:
                fpath = os.path.join(self.module_dir, "segmented-images", 
                                     os.path.splitext(self.image_name)[0] + "_masked_image.png")
            image.save(fpath)
        return transforms.functional.to_tensor(image)

    def create_dirs(self, root):
        """Creates directories used in segmentation
        Args:
            root (str): Root directory under which sub-directories be created
        """
        dir_names = ["segmented-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)

    def paste_multiple_masks_on_image(self, image, annotations, fpath=None):
        """Pastes multiple masks onto given image and saves"""
        print("Number of masks:", len(annotations))
        for annotation in sorted(annotations, key=lambda x: x["area"], reverse=True):
            image = self.paste_mask_on_image(image, annotation["segmentation"])
        
        if fpath is None:
            fpath = os.path.join(self.module_dir, "segmented-images", 
                                 os.path.splitext(self.image_name)[0] + "_multiple_masks_on_image.png")
        transforms.functional.to_pil_image(image).save(fpath)

    def get_mask_via_bbox_prompt(self, annotations, bbox_prompt):
        masks = torch.cat([annotation["segmentation"][None] for annotation in annotations])
        intersection = masks[:, bbox_prompt[1]:bbox_prompt[3], bbox_prompt[0]:bbox_prompt[2]].sum(dim=(1, 2))
        union = (bbox_prompt[0] + bbox_prompt[2])*(bbox_prompt[1] + bbox_prompt[3]) \
                + masks.sum(dim=(1, 2)) \
                - intersection
        iou_idx = torch.argmax(intersection/union)
        return masks[iou_idx]
    
    def annotations_to_masks(self, annotations):
        """Returns all masks in reverse sorted way for given annotations"""
        return [annotation["segmentation"] for annotation in sorted(annotations, key=lambda x: x["area"], reverse=True)]
    
    def aggregate_bbox_masks(self, annotations, bbox_prompts, label_prompts):
        """Returns aggregated mask for given bounding box and label prompts"""
        aggregated_mask = torch.zeros_like(annotations[0]["segmentation"])
        for bbox_prompt, label_prompt in zip(bbox_prompts, label_prompts):
            aggregated_mask[self.get_mask_via_bbox_prompt(annotations, bbox_prompt)] = 1 if label_prompt else 0
        return aggregated_mask == 1
    
    def plot_bbox_prompts_on_image(self, image, bbox_prompts, label_prompts=None, fpath=None):
        """Plots bounding box onto image and saves for given image, bbox, and label prompts"""
        if label_prompts is None:
            label_prompts = [1]*len(bbox_prompts)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0).numpy())
        for box, label in zip(bbox_prompts, label_prompts):
            self._plot_box(box, label, ax)
        ax.axis('on')

        if fpath is None:
            fpath = os.path.join(self.module_dir, "segmented-images", 
                                 os.path.splitext(self.image_name)[0] + "_bbox_prompts_on_image.png")
        fig.savefig(fpath)
        
    def _plot_box(self, box, label, ax):
        """Plots rectangle onto ax object for given bounding box and its label"""
        ax.add_patch(plt.Rectangle(
            (box[:2]),
            box[2] - box[0],
            box[3] - box[1],
            edgecolor = "green" if label else "red",
            facecolor = "none",
            lw = 2))


if __name__ == "__main__":

    prompt_sam = PromptSAM("dogs.jpg")
    bbox_prompts = [[200, 100, 600, 900], [200, 100, 500, 500]]
    label_prompts = [1, 0]
    prompt_sam.segment(bbox_prompts, label_prompts, (1024, 1024))