from argparse import ArgumentParser
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional, Union, Tuple

import cv2
import torch
import numpy as np

from yolact.yolact_net import Yolact
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers.output_utils import postprocess
from yolact.data import cfg, set_cfg


def clean_print(msg: str, fallback: Optional[Tuple[int, int]] = (156, 38), end='\n'):
    """ Function that prints the given string to the console and erases any previous print made on the same line

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


class YolactK:
    def __init__(self,
                 checkpoint_path: Path = "../../checkpoints/yolact_darknet53_9999_120000.pth",
                 config: str = "yolact_darknet53_config",
                 output_dir_path=None,
                 use_gpu=True,
                 verbose=False):
        """

        Args:
            checkpoint_path (Path): Path to the checkpoint to use.
            config (str): The config to use
            output_dir_path (Path, Optional):
            use_gpu (bool): Controls wether to use a gpu or not
            verbose (bool): If true then prints information useful for debugging
        """
        if output_dir_path:
            output_dir_path: Path = output_dir_path
            output_dir_path.mkdir(parents=True, exist_ok=True)
        self.output_dir_path = output_dir_path
        self.verbose = verbose

        # Set the yolact config and the default tensor type
        set_cfg(config)
        torch.set_default_tensor_type("torch.cuda.FloatTensor" if use_gpu else "torch.FloatTensor")

        print("Loading yolact model...", end="\r")
        self.net = Yolact()
        self.net.load_weights(str(checkpoint_path))
        self.net.eval()
        clean_print("Yolact model loaded")

        if use_gpu:
            self.net = self.net.cuda()

        # Use the default values
        self.net.detect.use_fast_nms = True  # Whether to use a faster, but not entirely correct version of NMS
        self.net.detect.use_cross_class_nms = False  # Whether compute NMS cross-class or per-class
        cfg.mask_proto_debug = False  # Outputs stuff for scripts/compute_mask.py

    def inference(self, imgs: np.ndarray, img_paths: Optional[Union[list[Path], Path]] = None) -> np.ndarray:
        """

        Args:
            imgs (np.ndarray): Either an image or a batch of images.
            img_paths (list, optional): If saving the result(s) as (an) image(s),
                                        should be the path(s) corresponding to the image(s).
        Returns:

        """
        # Handle case where input is not a batch
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, axis=0)
            img_paths = np.expand_dims(img_paths, axis=0) if img_paths else None
        if img_paths is None:
            img_paths = list(np.arange(len(imgs)))

        list_of_centers = []

        with torch.no_grad():
            for img, img_path in zip(imgs, img_paths):
                if self.output_dir_path:
                    assert img_path, "If saving results as images, the image path must be provided"
                    output_path = (self.output_dir_path / img_path.name).with_suffix(".png")

                # List that will store all the hair center for this image
                img_centers = []

                # Run image through the network
                img_gpu = torch.from_numpy(img).cuda().float()
                batch = FastBaseTransform()(img_gpu.unsqueeze(0))
                preds = self.net(batch)
                h, w, _ = img.shape

                # Post process
                t = postprocess(preds, w, h, visualize_lincomb=True, crop_masks=True, score_threshold=0.15)
                top_k = 15  # Further restrict the number of predictions to parse
                idx = t[1].argsort(0, descending=True)[:top_k]
                masks = t[3][idx].cpu().numpy()
                classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

                # Visualization
                for detection_index in range(len(boxes)):
                    # if self.verbose:
                    #     print(f"Number of non zero pixels in mask: {np.sum(masks[detection_index])}")

                    # Draw bounding box on the image
                    top_x, top_y, width, height = boxes[detection_index]
                    # Make sure that the bounding boxes are not going outside the image
                    width = min(width, img.shape[0] - top_x)
                    height = min(height, img.shape[1] - top_y)
                    center_x, center_y = top_x + width//2, top_y + height//2
                    if self.verbose:
                        print(f"bounging box center: {(center_x, center_y)}")
                    if self.output_dir_path:
                        img = cv2.rectangle(img, (top_x, top_y), (top_x+width, top_y+height), (0, 0, 255), 5)

                    if self.output_dir_path:
                        # Add the mask to the image
                        color = np.random.random(3)*255  # (0, 0, 255)
                        img[masks[detection_index].astype(np.bool8)] = color

                    # Get the point on the mask that is the closest to the bounding box's center
                    # Super inefficient but should work so....
                    dists = np.empty_like(masks[detection_index])
                    for i in range(dists.shape[0]):
                        for j in range(dists.shape[1]):
                            dists[i][j] = (center_y - i)**2 + (center_x - j)**2
                    dists = np.where(masks[detection_index], dists, np.inf)
                    closest_point = np.unravel_index(dists.argmin(), dists.shape)
                    if self.verbose:
                        print(f"{closest_point=}")

                    # If the bounding box's center is already on the mask, then just return that
                    if closest_point == (center_y, center_x):
                        if self.verbose:
                            print("Bounding box's center was already on the mask")
                        center = end_point = closest_point
                    # Otherwise, "trace" a line going through the bounding box's center and the closest point on the
                    # mask. We start inspecting that line from the closest point and in the direction going away from
                    # the bounding box's center. Stops when the line goes out of the mask.
                    else:
                        # Get unit vector
                        line_dir = np.asarray([closest_point[0]-center_y, closest_point[1]-center_x])
                        if self.verbose:
                            print(f"line_dir before normalization: {line_dir}")
                        line_dir = line_dir / np.amax(np.abs(line_dir))
                        if self.verbose:
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

                    if self.verbose:
                        print(f"{center=}")

                    point_size = 3
                    if self.output_dir_path:
                        # Add closest point on image
                        img = cv2.circle(img, (closest_point[1], closest_point[0]), point_size, (0, 255, 0), point_size)
                        # Add line end point
                        img = cv2.circle(img, (end_point[1], end_point[0]), point_size, (255, 255, 255), point_size)
                        # Add found center
                        img = cv2.circle(img, (center[1], center[0]), point_size, (255, 0, 0), point_size)
                        # Add bounding box center
                        img = cv2.circle(img, (center_x, center_y), point_size, (0, 0, 0), point_size)

                    img_centers.append(center)

                if self.output_dir_path:
                    cv2.imwrite(str(output_path), img)
                list_of_centers.append(img_centers)

        return list_of_centers


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

    yolact_k = YolactK(checkpoint_path=args.trained_model, config=args.config, output_dir_path=args.output_dir_path,
                       use_gpu=args.use_gpu, verbose=args.verbose)
    exts = [".jpg", ".png"]
    img_paths = list([p for p in data_path.rglob("*") if p.suffix in exts])
    nb_imgs = len(img_paths)
    for img_index, img_path in enumerate(img_paths):
        clean_print(f"Processing image: {img_path} ({img_index+1}/{nb_imgs})", end="\n" if args.verbose else "\r")
        img = cv2.imread(str(img_path))
        centers = yolact_k.inference(img, img_path)
        if args.verbose:
            print(f"Centers for img {img_path}: {centers}")


if __name__ == "__main__":
    main()
