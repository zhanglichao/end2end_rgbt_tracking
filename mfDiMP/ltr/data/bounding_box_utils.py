import numpy as np
import torch
import random
import math
from random import gauss


def calc_iou(box_a: np.array, box_b: np.array) -> float:
    """ Calculates IoU overlap of every box in box_a with the corresponding box in box_b.
    args:
        box_a - numpy array of shape [x1, x2, x3 .....,, xn, 4]
        box_b - numpy array of shape [x1, x2, x3 .....,, xn, 4]

    returns:
        np.array - array of shape [x1, x2, x3 .....,, xn], containing IoU overlaps
    """

    eps = 1e-10

    x1 = np.maximum(box_a[..., 0], box_b[..., 0])
    y1 = np.maximum(box_a[..., 1], box_b[..., 1])
    x2 = np.minimum(box_a[..., 0] + box_a[..., 2], box_b[..., 0] + box_b[..., 2])
    y2 = np.minimum(box_a[..., 1] + box_a[..., 3], box_b[..., 1] + box_b[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    area_intersect = w*h
    area_a = box_a[..., 2] * box_a[..., 3]
    area_b = box_b[..., 2] * box_b[..., 3]

    area_overlap = area_a + area_b - area_intersect + eps

    iou = area_intersect / area_overlap
    return iou


def transform_image_to_crop(box_in: np.array, box_extract: np.array, resize_factor: float,
                            crop_sz: np.array) -> np.array:
    box_extract_center = box_extract[0:2] + 0.5*box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5*box_in[2:4]

    box_out_center = (crop_sz-1)/2 + (box_in_center - box_extract_center)*resize_factor
    box_out_wh = box_in[2:4]*resize_factor

    box_out = np.concatenate((box_out_center - 0.5*box_out_wh, box_out_wh))
    return box_out


def fit_inside_image(box: np.array, limit: np.array) ->np.array:
    box[0] = min(max(box[0], limit[0]), limit[2] - box[2])
    box[1] = min(max(box[1], limit[1]), limit[3] - box[3])
    box[2] = max(min(box[0] + box[2], limit[2] - 1), limit[0]) - box[0]
    box[3] = max(min(box[1] + box[3], limit[3] - 1), limit[1]) - box[1]

    return box


def perturb_box(box: np.array, min_iou=0.5, sigma_factor=0.1, p_ar_jitter=None, p_scale_jitter=None,
                use_gaussian=False, sig=None):
    """ Clean this up!!!"""
    if isinstance(sigma_factor, list):
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, np.ndarray):
        c_sigma_factor = c_sigma_factor * np.ones(4)

    ar_jitter = False
    scale_jitter = False

    if p_ar_jitter is not None and random.uniform(0, 1) < p_ar_jitter:
        ar_jitter = True
    elif p_scale_jitter is not None and random.uniform(0, 1) < p_scale_jitter:
        scale_jitter = True

    perturb_factor = np.sqrt(box[2]*box[3])*c_sigma_factor

    if ar_jitter or scale_jitter:
        perturb_factor[0:2] = np.sqrt(box[2]*box[3])*0.05

    for i_ in range(100):
        c_x = box[0] + 0.5*box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = gauss(c_x, perturb_factor[0])
        c_y_per = gauss(c_y, perturb_factor[1])

        if p_scale_jitter:
            w_per = gauss(box[2], perturb_factor[2])
            h_per = box[3] * w_per / (box[2] + 1)
        else:
            w_per = gauss(box[2], perturb_factor[2])
            h_per = gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2]*np.random.uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3]*np.random.uniform(0.15, 0.5)

        box_per = np.array([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2]*np.random.uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3]*np.random.uniform(0.15, 0.5)

        iou = calc_iou(box, box_per)

        if iou > min_iou:
            return box_per, iou, 0

        # Reduce perturb factor
        perturb_factor *= 0.9

    return box_per, iou, 1
