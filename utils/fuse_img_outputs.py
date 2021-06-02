import numpy as np
import cv2


def get_IOU(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """ Computes the IOU between two masks (assuming masks are 1s and 0s) """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1 + mask2, dtype=np.float32)
    iou = intersection / union
    return iou


def fuse_outputs(masks_list: list[np.ndarray], iou_threshold: float = 0.0001) -> np.ndarray:
    """ Fuse the predictions made on several frames of a video.

    Args:
        masks_list (list): List with the masks prediction for each frame.
                           Each prediction is expected to have shape (N, width, height),
                           with N being the number of objects detected on that frame,
                           width and height the size of the image.

    Returns:
        np.ndarray: Fused prediction, has shape (N, width, height)
    """
    # # Get average fiber size for each image
    # avg_sizes = [np.sum(frame_masks) / len(frame_masks) for frame_masks in masks_list]
    # # Give an id to each fiber, by starting with the image with highest average fiber size.
    # start_frame_idx = np.argmax(avg_sizes)

    # Start with the frame with the least detections (Exluding frames with no detections)
    nb_detected_per_frame = np.asarray([len(frame_masks) for frame_masks in masks_list])
    valid_idx = np.where(nb_detected_per_frame > 0)[0]
    start_frame_idx = valid_idx[nb_detected_per_frame[valid_idx].argmin()]

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
                np.append(fused_masks, mask1)
            # Otherwise fuse the 2 masks that matched
            else:
                fused_masks[best_match_idx] = np.clip(fused_masks[best_match_idx] + mask1, 0, 1)

    # Compute the new bounding boxes
    bboxes = []
    for mask in fused_masks:
        mask = np.expand_dims(mask, -1).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [cv2.boundingRect(contour) for contour in contours]
        rect = rectangles[0]

        # Quick fix in case the mask is not a continuous blob
        if len(rectangles) > 1:
            for i in range(1, len(rectangles)):
                rect = merge_rect(rect, rectangles[i])

        x, y, w, h = rect
        bboxes.append([int(x), int(y), int(w), int(h)])

    return fused_masks, bboxes


def merge_rect(a, b):
    """ Returns the union of two rectangles """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)
