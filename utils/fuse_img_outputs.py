import numpy as np
import cv2


def get_IOU(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """ Computes the IOU between two masks (assuming masks are 1s and 0s) """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1 + mask2, dtype=np.float32)
    iou = intersection / union
    return iou


def fuse_outputs(masks_list: list[np.ndarray], iou_threshold: float = 0.1) -> np.ndarray:
    """ Fuse the predictions made on several frames of a video.

    Args:
        masks_list (list): List with the masks prediction for each frame.
                           Each prediction is expected to have shape (N, width, height),
                           with N being the number of objects detected on that frame,
                           width and height the size of the image.

    Returns:
        np.ndarray: Fused prediction, has shape (N, width, height)
    """
    # fused_masks = np.zeros(masks_list[0][0].shape)   # (550, 550)
    # max_nb_fibers = max([len(output) for output in masks_list])

    # Get average fiber size for each image
    avg_sizes = [np.sum(frame_masks) / len(frame_masks) for frame_masks in masks_list]

    # Give an id to each fiber, by starting with the image with highest average fiber size.
    start_frame_idx = np.argmax(avg_sizes)
    fused_masks = masks_list[start_frame_idx]

    # For all the other images, find which fiber they match best and if needed add an id
    for frame_idx in range(len(masks_list)):
        if frame_idx == start_frame_idx:
            continue
        for mask1 in masks_list[frame_idx]:
            iou_list = [get_IOU(mask1, mask2) for mask2 in fused_masks]
            best_match_idx = np.argmax(iou_list)
            # If the best IOU is bellow the threshold, then we've found a new object
            if iou_list[best_match_idx] < iou_threshold:
                fused_masks.append(mask1)
            # Otherwise fuse the 2 masks that matched
            else:
                fused_masks[best_match_idx] += mask1

    # Compute the new bounding boxes
    bboxes = []
    for mask in fused_masks:
        mask = np.expand_dims(mask, -1).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        bboxes.append([int(x), int(y), int(w), int(h)])

    return fused_masks, bboxes
