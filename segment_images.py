#!/usr/bin/env python
# manual

"""
Script to generate segmentation labels with trained BiSeNet
"""

import sys
import os
import argparse
import numpy as np

from PIL import Image
from glob import glob


from duckienet_config import bisenet_cfg

import torch
import torch.nn as nn

sys.path.append('/home/yuanbo/projects/2309/bisenet')
from lib.models import BiSeNetV2
import lib.transform_cv2 as T


torch.set_grad_enabled(False)

# Load bisenet model
bisenet = BiSeNetV2(bisenet_cfg['n_classes'])
bisenet.load_state_dict(torch.load(bisenet_cfg['weights_path'], map_location='cpu'))
bisenet.eval()
bisenet.cuda()

# prepare data (for bisenet)
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

# Set up directories
images_dir = "/home/yuanbo/projects/2309/gym-duckietown/data/images"
labels_dir = "/home/yuanbo/projects/2309/gym-duckietown/data/labels"

# Retrieve list of images
list_images = glob(f"{images_dir}/X_*")

for i in range(len(list_images)):
    img = np.array(Image.open(list_images[i]))
    
    # Generate segmentation labels from using bisenet
    img = to_tensor(dict(im=img, lb=None))['im'].unsqueeze(0).cuda()
    labels = bisenet(img)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    labels = labels.astype(np.uint8)
    labels = Image.fromarray(labels)

    # Save image
    labels.save(labels_dir + f"/L_{i}.png")

print(f"Segmented {len(list_images)} images!")
