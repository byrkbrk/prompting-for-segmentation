# Prompting for Segmentation

## Introduction

We implement a module for segmenting images using (points and bounding box) prompts, in PyTorch. As a model, we use pretrained [FastSAM](https://arxiv.org/abs/2306.12156) provided by [ultralytics](https://docs.ultralytics.com/models/fast-sam/). 

## Prompts

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