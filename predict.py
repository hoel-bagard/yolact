from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional, Tuple

import cv2
import torch
import numpy as np

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from data import cfg, set_cfg


def clean_print(msg: str, fallback: Optional[Tuple[int, int]] = (156, 38), end='\n'):
    """ Function that prints the given string to the console and erases any previous print made on the same line

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


def main():
    parser = ArgumentParser("Tool to visualize coco labels.")
    parser.add_argument("data_path", type=Path, help="Path to the directory with the images to process")
    parser.add_argument("config", help="The config object to use.")
    parser.add_argument("trained_model", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("--output_dir_path", "--o", type=Path, default=None, help="Path to an output dir to get images")
    parser.add_argument("--use_gpu", "--gpu", action="store_true", help="Use cuda.")
    parser.add_argument("--verbose", "--v", action="store_true", help="Use for debug.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    if args.output_dir_path:
        output_dir_path: Path = args.output_dir_path
        output_dir_path.mkdir(parents=True, exist_ok=True)

    list_of_centers = []

    set_cfg(args.config)
    torch.set_default_tensor_type("torch.cuda.FloatTensor" if args.use_gpu else "torch.FloatTensor")

    with torch.no_grad():
        print("Loading model...", end="\r")
        net = Yolact()
        net.load_weights(str(args.trained_model))
        net.eval()
        clean_print("Model loaded")

        if args.use_gpu:
            net = net.cuda()

        # Use the default values
        net.detect.use_fast_nms = True  # Whether to use a faster, but not entirely correct version of NMS
        net.detect.use_cross_class_nms = False  # Whether compute NMS cross-class or per-class
        cfg.mask_proto_debug = False  # Outputs stuff for scripts/compute_mask.py

        img_paths = list(data_path.rglob("*.jpg"))
        nb_imgs = len(img_paths)
        for img_index, img_path in enumerate(img_paths):
            clean_print(f"Processing image: {img_path} ({img_index+1}/{nb_imgs})", end="\n" if args.verbose else "\r")
            if args.output_dir_path:
                output_path = (output_dir_path / img_path.name).with_suffix(".png")

            # List that will store all the hair center for this image
            img_centers = []

            img = torch.from_numpy(cv2.imread(str(img_path))).cuda().float()
            batch = FastBaseTransform()(img.unsqueeze(0))
            preds = net(batch)
            img_gpu = img / 255.0
            h, w, _ = img.shape

            t = postprocess(preds, w, h, visualize_lincomb=True, crop_masks=True, score_threshold=0.15)
            top_k = 15  # Further restrict the number of predictions to parse
            idx = t[1].argsort(0, descending=True)[:top_k]
            masks = t[3][idx].cpu().numpy()
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

            # Visualization
            img = cv2.imread(str(img_path))
            for detection_index in range(len(boxes)):
                # if args.verbose:
                #     print(f"Number of non zero pixels in mask: {np.sum(masks[detection_index])}")

                # Draw bounding box on the image
                top_x, top_y, width, height = boxes[detection_index]
                # Make sure that the bounding boxes are not going outside the image
                width = min(width, img.shape[0] - top_x)
                height = min(height, img.shape[1] - top_y)
                center_x, center_y = top_x + width//2, top_y + height//2
                if args.verbose:
                    print(f"bounging box center: {(center_x, center_y)}")
                if args.output_dir_path:
                    img = cv2.rectangle(img, (top_x, top_y), (top_x+width, top_y+height), (0, 0, 255), 5)

                if args.output_dir_path:
                    # Add the mask to the image
                    # img[masks[detection_index].astype(np.uint8)] = (0, 0, 255)
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            if masks[detection_index, i, j]:
                                img[i][j] = (0, 0, 255)

                # Get the point on the mask that is the closest to the bounding box's center
                # Super inefficient but should work so....
                dists = np.empty_like(masks[detection_index])
                for i in range(dists.shape[0]):
                    for j in range(dists.shape[1]):
                        dists[i][j] = (center_y - i)**2 + (center_x - j)**2
                dists = np.where(masks[detection_index], dists, np.inf)
                closest_point = np.unravel_index(dists.argmin(), dists.shape)
                if args.verbose:
                    print(f"{closest_point=}")

                # If the bounding box's center is already on the mask, then just return that
                if closest_point == (center_y, center_x):
                    if args.verbose:
                        print("Bounding box's center was already on the mask")
                    center = end_point = closest_point
                # Otherwise, "trace" a line going through the bounding box's center and the closest point on the
                # mask. We start inspecting that line from the closest point and in the direction going away from the
                # bounding box's center. Stops when the line goes out of the mask.
                else:
                    # Get unit vector
                    line_dir = np.asarray([closest_point[0]-center_y, closest_point[1]-center_x])
                    if args.verbose:
                        print(f"line_dir before normalization: {line_dir}")
                    line_dir = line_dir / np.amax(np.abs(line_dir))
                    if args.verbose:
                        print(f"Normalized line_dir: {line_dir}")
                    mult_factor = 1
                    current_x = closest_point[0] + int(line_dir[0] * mult_factor)
                    current_y = closest_point[1] + int(line_dir[1] * mult_factor)
                    while (0 <= current_x < img.shape[0]
                           and 0 <= current_y < img.shape[1]
                           and masks[detection_index, current_x, current_y]):
                        current_x = closest_point[0] + int(line_dir[0] * mult_factor)
                        current_y = closest_point[1] + int(line_dir[1] * mult_factor)
                        mult_factor += 1

                    center = closest_point + (line_dir * mult_factor/2).astype(int)
                    end_point = closest_point + (line_dir * mult_factor).astype(int)

                if args.verbose:
                    print(f"{center=}")

                point_size = 3
                if args.output_dir_path:
                    # Add closest point on image
                    img = cv2.circle(img, (closest_point[1], closest_point[0]), point_size, (0, 255, 0), point_size)
                    # Add line end point
                    img = cv2.circle(img, (end_point[1], end_point[0]), point_size, (255, 255, 255), point_size)
                    # Add found center
                    img = cv2.circle(img, (center[1], center[0]), point_size, (255, 0, 0), point_size)
                    # Add bounding box center
                    img = cv2.circle(img, (center_x, center_y), point_size, (0, 0, 0), point_size)

                img_centers.append(center)

            if args.output_dir_path:
                cv2.imwrite(str(output_path), img)
            list_of_centers.append(img_centers)

            # if img_index >= 5:
            #     break
    print("\nResult: ")
    print(list_of_centers)


if __name__ == "__main__":
    main()
