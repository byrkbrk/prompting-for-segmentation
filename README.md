# Prompting for Segmentation

## Introduction

We implement a module for segmenting images using (points and bounding box) prompts, in PyTorch. As a model, we use pretrained [FastSAM](https://arxiv.org/abs/2306.12156) provided by [ultralytics](https://docs.ultralytics.com/models/fast-sam/). 

## Setting Up the Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/prompting-for-segmentation.git
    ~~~
3. Change the directory:
    ~~~
    cd prompting-for-segmentation
    ~~~
4. For macos, run:
    ~~~
    conda env create -f diffusion-env_macos.yaml
    ~~~
    For linux or windows, run:
    ~~~
    conda env create -f diffusion-env_linux.yaml
    ~~~
5. Activate the environment:
    ~~~
    conda activate diffusion-env
    ~~~

## Prompts

Check it out how to use:

~~~
python3 segment.py -h
~~~

Output:

~~~
Segment given image

positional arguments:
  image_name            Name of the image that be processed
  point_or_bbox_prompts
                        List of point prompts in (height, width) or bbox prompts in (x_min, y_min, x_max, y_max)

options:
  -h, --help            show this help message and exit
  --label_prompts LABEL_PROMPTS [LABEL_PROMPTS ...]
                        List of labels of point prompts
  --image_size IMAGE_SIZE [IMAGE_SIZE ...]
                        Size (height, width) to which the image be transformed
  --checkpoint_name {FastSAM-x.pt,FastSAM-s.pt}
                        Name of the pretrained model for FastSAM
  --device DEVICE       Name of the device on which the model be run
~~~



### Point prompts

#### Using single prompt

~~~
python3 segment.py dogs.jpg 400,400 --label_prompts 1 --image_size 1024 1024
~~~

<p align="center">
  <img src="files-for-readme/dogs_multiple_masks_on_image.png" width="30%" />
  <img src="files-for-readme/dogs_point_prompts_on_image_1.png" width="30%" />
  <img src="files-for-readme/dogs_masked_image_via_points_1.png" width="30%" />
</p>

#### Using multiple prompts

~~~
python3 segment.py dogs.jpg 400,400 700,400 --label_prompts 0 1 --image_size 1024 1024
~~~

<p align="center">
  <img src="files-for-readme/dogs_multiple_masks_on_image.png" width="30%" />
  <img src="files-for-readme/dogs_point_prompts_on_image_2.png" width="30%" />
  <img src="files-for-readme/dogs_masked_image_via_points_2.png" width="30%" />
</p>


### Bounding box prompts

#### Using single prompt

~~~
python3 segment.py dogs.jpg 625,625,700,700 --label_prompts 1 --image_size 1024 1024
~~~


<p align="center">
  <img src="files-for-readme/dogs_multiple_masks_on_image.png" width="30%" />
  <img src="files-for-readme/dogs_bbox_prompts_on_image_1.png" width="30%" />
  <img src="files-for-readme/dogs_masked_image_via_bboxes_1.png" width="30%" />
</p>

#### Using multiple prompts

~~~
python3 segment.py dogs.jpg 500,200,800,900 510,210,810,610 --label_prompts 1 0 --image_size 1024 1024
~~~

<p align="center">
  <img src="files-for-readme/dogs_multiple_masks_on_image.png" width="30%" />
  <img src="files-for-readme/dogs_bbox_prompts_on_image_2.png" width="30%" />
  <img src="files-for-readme/dogs_masked_image_via_bboxes_2.png" width="30%" />
</p>