import os.path as osp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from core.config import cfg

from coord_utils import get_max_preds

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        if kps[i][2] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints_with_skeleton(img, kps, kps_line, bbox=None, kp_thre=0.4, alpha=1):
    # Convert form plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_line))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    
    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)
    
    # Perfrom the drawing on a copy of the image, to allow for blending
    kp_mask = np.copy(img)

    # Draw bounding box
    if bbox is not None:
        b1 = bbox[0, 0].astype(np.int32), bbox[0, 1].astype(np.int32)
        b2 = bbox[1, 0].astype(np.int32), bbox[1, 1].astype(np.int32)
        b3 = bbox[2, 0].astype(np.int32), bbox[2, 1].astype(np.int32)
        b4 = bbox[3, 0].astype(np.int32), bbox[3, 1].astype(np.int32)

        cv2.line(kp_mask, b1, b2, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b2, b3, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b3, b4, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(kp_mask, b4, b1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw the keypoints
    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        
        p1 = kps[i1,0].astype(np.int32), kps[i1,1].astype(np.int32)
        p2 = kps[i2,0].astype(np.int32), kps[i2,1].astype(np.int32)
        if kps[i1,2] > kp_thre and kps[i2,2] > kp_thre:
            cv2.line(
                kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[i1,2] > kp_thre:
            cv2.circle(
                kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[i2,2] > kp_thre:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_2d_pose(pred, img, kps_line, prefix='vis2dpose', bbox=None):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    tmpimg = img.copy().astype(np.uint8)
    tmpkps = np.zeros((3, len(pred)))
    tmpkps[0, :], tmpkps[1, :] = pred[:, 0], pred[:, 1]
    tmpkps[2, :] = 1
    tmpimg = vis_keypoints_with_skeleton(tmpimg, tmpkps, kps_line, bbox)

    now = datetime.now()
    file_name = f'{prefix}_{now.isoformat()[:-7]}_2d_joint.jpg'
    cv2.imwrite(osp.join(cfg.vis_dir, file_name), tmpimg)
    #cv2.imshow(prefix, tmpimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def vis_3d_pose(kps_3d, kps_line, joint_set_name='', file_path='image.png', ax_in=None, kps_3d_vis=None):
    if joint_set_name == 'human36':
        r_joints = [1, 2, 3, 14, 15, 16]
    elif joint_set_name == 'coco':
        r_joints = [2, 4, 6, 8, 10, 12, 14, 16]
    elif joint_set_name == 'smpl':
        r_joints = [2, 5, 8, 11, 14, 17, 19, 21, 23]
    else:
        r_joints = []

    if kps_3d_vis is None:
        kps_3d_vis = np.ones((len(kps_3d), 1))
    else:
        kps_3d_vis = kps_3d_vis.reshape(-1, 1)
        
    if not ax_in:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax_in

    for l in range(len(kps_line)):
        i1 = kps_line[l][0]
        i2 = kps_line[l][1]
        x = np.array([kps_3d[i1, 0], kps_3d[i2, 0]])
        y = np.array([kps_3d[i1, 1], kps_3d[i2, 1]])
        z = np.array([kps_3d[i1, 2], kps_3d[i2, 2]])

        if kps_3d_vis[i1, 0] > 0 and kps_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c='r', linewidth=1)
        if kps_3d_vis[i1, 0] > 0:
            c = 'g' if i1 in r_joints else 'b'
            ax.scatter(kps_3d[i1, 0], kps_3d[i1, 2], -kps_3d[i1, 1], c=c, marker='o')
        if kps_3d_vis[i2, 0] > 0:
            c = 'g' if i2 in r_joints else 'b'
            ax.scatter(kps_3d[i2, 0], kps_3d[i2, 2], -kps_3d[i2, 1], c=c, marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')
    
    ax.set_xlim3d(-800, 800)
    ax.set_ylim3d(-800, 800)
    ax.set_zlim3d(-800, 800)

    title = '3D Skeleton'
    ax.set_title(title)
    axisEqual3D(ax)

    if not ax_in:
        fig.savefig(file_path)
        plt.close(fig=fig)
    else:
        return ax

def save_obj(v, f=None, file_name=''):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def plot_joint_error(mpjpe, mpjve, mpjae):
    mpjae = np.concatenate((mpjae,np.zeros((1,))))

    f = plt.figure()
    plot_title = 'MPJPE'
    file_ext = '.jpg'
    save_path = '_'.join(plot_title.split(' ')).lower() + file_ext
    plt.plot(np.arange(1, len(mpjpe) + 1), mpjpe, 'b-', label='MPJPE')
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('frame')
    plt.xlim(left=0, right=len(mpjpe) + 1)
    plt.xticks(np.arange(0, len(mpjpe) + 1, 50.0), fontsize=5)
    f.savefig(os.path.join(cfg.graph_dir, save_path))
    plt.close(f)

    f = plt.figure()
    plot_title = 'MPJVE & MPJAE'
    file_ext = '.jpg'
    save_path = '_'.join(plot_title.split(' ')).lower() + file_ext
    plt.plot(np.arange(1, len(mpjve) + 1), mpjve, 'b-', label='MPJVE')
    plt.plot(np.arange(1, len(mpjae) + 1), mpjae, 'r-', label='MPJAE')
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('frame')
    plt.xlim(left=0, right=len(mpjve) + 1)
    plt.xticks(np.arange(0, len(mpjve) + 1, 50.0), fontsize=5)
    f.savefig(os.path.join(cfg.graph_dir, save_path))
    plt.close(f)


def save_plot(data_list, epoch, title='Train Loss', show_min=True):
    f = plt.figure()

    plot_title = '{} epoch {}'.format(title, epoch)
    file_ext = '.pdf'
    save_path = '_'.join(title.split(' ')).lower() + file_ext

    plt.plot(np.arange(1, len(data_list) + 1), data_list, 'b-', label=plot_title)
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.xlim(left=0, right=len(data_list) + 1)
    plt.xticks(np.arange(0, len(data_list) + 1, 1.0), fontsize=5)

    if show_min:
        min_value = np.asarray(data_list).min()
    else:
        min_value = np.asarray(data_list).max()

    plt.annotate('%0.2f' % min_value, xy=(1, min_value), xytext=(8, 0),
                 arrowprops=dict(arrowstyle="simple", connectionstyle="angle3"),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    f.savefig(os.path.join(cfg.graph_dir, save_path))
    plt.close(f)
    

def vis_heatmaps(batch_image, batch_heatmaps):
    '''
    batch_image: [batch_size, height, width, channel]
    batch_heatmaps: [batch_size, num_joints, height, width]
    '''
    batch_image = batch_image.astype(np.uint8)
    batch_heatmaps = (batch_heatmaps * 255).astype(np.uint8)
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)
    
    preds, maxvals = get_max_preds(batch_heatmaps)

    radius = math.ceil(batch_heatmaps.shape[0] / 64)

    for i in range(batch_size):
        image = batch_image[i]
        heatmaps = batch_heatmaps[i]

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       radius, [0, 0, 255], -1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    return grid_image