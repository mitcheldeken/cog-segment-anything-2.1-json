# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import cv2
import sys
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List
import json
# Add /tmp/sa2 to sys path
sys.path.extend("/sa2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

WEIGHTS_CACHE = "checkpoints"
MODEL_NAME = "sam2_hiera_large.pt"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["wget", "-O", dest, url], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.chdir("/sa2")
        # Get path to model
        model_cfg = "sam2_hiera_l.yaml"
        model_path = WEIGHTS_CACHE + "/" +MODEL_NAME
        # Download weights
        if not os.path.exists(model_path):
            download_weights(WEIGHTS_URL, model_path)
        # Setup SAM2
        self.sam2 = build_sam2(config_file=model_cfg, ckpt_path=model_path, device='cuda', apply_postprocessing=False)
        # turn on tfloat32 for Ampere GPUs
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask_limit: int = Input(
            default=-1, description="maximum number of masks to return. If -1 or None, all masks will be returned. NOTE: The masks are sorted by predicted_iou."),
        points_per_side: int = Input(
            default=64, description="The number of points to be sampled along one side of the image."),
        points_per_batch: int = Input(
            default=128, description="Sets the number of points run simultaneously by the model"),
        pred_iou_thresh: float = Input(
            default=0.7, description="A filtering threshold in [0,1], using the model's predicted mask quality."),
        stability_score_thresh: float = Input(
            default=0.92, description="A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions."),
        stability_score_offset: float = Input(
            default=0.7, description="The amount to shift the cutoff when calculated the stability score."),
        crop_n_layers: int = Input(
            default=1, description="If >0, mask prediction will be run again on crops of the image"),
        box_nms_thresh: float = Input(
            default=0.7, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks."),
        crop_n_points_downscale_factor: int = Input(
            default=2, description="The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n."),
        min_mask_region_area: float = Input(
            default=25.0, description="If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area."),
        mask_2_mask: bool = Input(
            default=True, description="Whether to add a one step refinement using previous mask predictions."),
        multimask_output: bool = Input(
            default=False, description="Whether to output multimask at each point of the grid."),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Convert input image
        image_rgb = Image.open(image).convert('RGB')
        image_arr = np.array(image_rgb)

        # Setup the predictor and image
        mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            use_m2m=mask_2_mask,
            multimask_output=multimask_output,
            output_mode="uncompressed_rle",
        )
        sam_output = mask_generator.generate(image_arr)
        with open('masks.json', 'w') as file:
            json.dump(sam_output, file)

        return Path('masks.json')
