import numpy as np
import torch
import math
import cv2
from torch.nn import functional as F
from core.config import cfg


def add_pelvis_and_neck(joint_coord, joint_valid, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    
    if joint_valid[lhip_idx] > 0 and joint_valid[rhip_idx] > 0:
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))
        joint_valid = np.append(joint_valid, 1)
    else:
        pelvis = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')

    if joint_valid[lshoulder_idx] > 0 and joint_valid[rshoulder_idx] > 0:
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))
        joint_valid = np.append(joint_valid, 1)
    else:
        neck = np.zeros_like(joint_coord[0, None, :])
        joint_valid = np.append(joint_valid, 0)

    joint_coord = np.concatenate((joint_coord, pelvis, neck))
    return joint_coord, joint_valid

def get_center_scale(box_info):
        x, y, w, h = box_info

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale = np.array([
            w * 1.0, h * 1.0
        ], dtype=np.float32)

        return center, scale

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height, do_sanitize=True):
    if do_sanitize:
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.MODEL.input_img_shape[1] / cfg.MODEL.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def pixel2cam(coords, c, f):
    cam_coord = np.zeros((len(coords), 3))
    z = coords[..., 2].reshape(-1, 1)

    cam_coord[..., :2] = (coords[..., :2] - c) * z / f
    cam_coord[..., 2] = coords[..., 2]
    return cam_coord

def generate_joint_heatmap(joints, joints_vis, image_size, heatmap_size, sigma=2):
    num_joints = joints.shape[0]
    image_size, heatmap_size = np.array(image_size[::-1]), np.array(heatmap_size[::-1])
    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    target_weight = joints_vis[:,None]
    
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight.reshape(-1)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def heatmap_to_coords(batch_heatmaps):
    coords, maxvals = get_max_preds(batch_heatmaps)
    maxvals = maxvals.reshape(maxvals.shape[0], maxvals.shape[1])
    
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px]-hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25
    
    coords = coords.copy()
    coords_valid = (maxvals > 0.5) * 1.0
    return coords, coords_valid

def image_bound_check(coord, image_size, val=None):
    if val is None:
        val = np.ones((len(coord),))
    else:
        val = val.copy()
    
    idxs = np.logical_or(coord[:,0] < 0, coord[:,0] > image_size[1])
    val[idxs] = 0
    
    idxs = np.logical_or(coord[:,1] < 0, coord[:,1] > image_size[0])
    val[idxs] = 0
    return val
    

def sampling_non_joint(hm, non_joint_num, neg_thr=0.5):
    joint_hm = (hm.sum(0) < neg_thr)
    idx = np.where(joint_hm)
    idx = np.stack(idx, axis=1)
    idx = idx[np.random.choice(len(idx), non_joint_num)]
    
    sampling_non_joints = np.concatenate([idx[:,1,None],idx[:,0,None]], axis=1)
    return sampling_non_joints.astype(np.float32)


def sampling_joint_coords(hm, joint_valid, non_joint_num, pos_thr=0.75, neg_thr=0.25):
    joint_num = joint_valid.shape[0]
    sampling_joints = np.zeros((joint_num, 2))
    
    joint_hm = (hm > pos_thr)
    for i in range(joint_num):
        if joint_valid[i] == 0:
            continue
        
        idx = np.where(joint_hm[i])
        idx = np.stack([idx[1],idx[0]], axis=1)
        
        if len(idx) == 0:
            joint_valid[i] = 0
            continue
        
        idx = idx[np.random.choice(len(idx), 1)[0]]
        sampling_joints[i] = idx
        
    joint_hm = (hm.sum(0) < neg_thr)
    idx = np.where(joint_hm)
    idx = np.stack(idx, axis=1)
    sampling_non_joints = idx[np.random.choice(len(idx), non_joint_num)]
    
    return sampling_joints, sampling_non_joints, joint_valid