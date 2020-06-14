from pathlib import Path
from scipy.spatial import Delaunay
import parseTrackletXML as ParserTrackletXML
import pykitti
import yaml
import numpy as np
from tqdm import tqdm

raw_kitti_base_path = Path('/usr/src/research/raw_data_downloader')
raw_kitti_config_path = Path('/usr/src/research/lidar-bonnetal/train/tasks/semantic/config/labels/raw-kitti.yaml')


def load_tracklet(tracklet_labels_file_path):
    """
    function read tracklet_labels file and returns annotation bounding boxes and labels

    Modified copy from: 
        https://github.com/windowsub0406/KITTI_Tutorial/blob/a53c99b7ba301b78f125ad075de115803b7f43a6/kitti_foundation.py

    Args: 
        tracklet_labels_file_path, Path, path to tracklet_labels file

    returns:
        frames_bouding_boxes, dictionary, key: frame_number -> value: list of numpy arrays of bouding box in velodyne space
        frames_bounding_boxes_labels, dictionary, key: frame_number -> value: list of KITTI annotation label for every box
    """

    # read info from xml file
    tracklets = ParserTrackletXML.parseXML(tracklet_labels_file_path)

    frames_bouding_boxes = {}
    frames_bounding_boxes_labels = {}

    # refered to parseTrackletXML.py's example function
    # loop over tracklets
    for tracklet in tracklets:

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absolute_frame_number in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (ParserTrackletXML.TRUNC_IN_IMAGE, ParserTrackletXML.TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are 0 in all xml files I checked
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([ \
                [np.cos(yaw), -np.sin(yaw), 0.0], \
                [np.sin(yaw), np.cos(yaw), 0.0], \
                [0.0, 0.0, 1.0]])

            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

            if absolute_frame_number in frames_bouding_boxes:
                frames_bouding_boxes[absolute_frame_number] += [cornerPosInVelo.T]
                frames_bounding_boxes_labels[absolute_frame_number] += [tracklet.objectType]
            else:
                frames_bouding_boxes[absolute_frame_number] = [cornerPosInVelo.T]
                frames_bounding_boxes_labels[absolute_frame_number] = [tracklet.objectType]


    return frames_bouding_boxes, frames_bounding_boxes_labels


def in_hull(points, hull):
    """
    function check if array of points is in hull

    Modified copy from:
        https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/prepare_data.py

    Args:
        points, numpy array, shape: (N,3), coordinates of point in velodyne space
        hull, numpy array, shape: (8,3), coordinates of corners of 3d bounding box

    returns:
        array of boolean for each point, true if point is in hull 
    """
    assert hull.shape[0] == 8, f"bounding box with few points expected 8, obtained {hull.shape}"
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(points)>=0

def get_idx_points_in_box3d(point_cloud, bouding_box_3D):
    """
    function gets the idx points that are inside bouding box
    
    Modified copy from:
        https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/prepare_data.py

    Args:
        point_cloud, numpy array, shape: (N, 3), LiDAR point cloud
        bouding_box_3D, numpy array, shape: (8, 3), coordinates of corners of 3D bouding box
    
    returns:
        idx of points inside bouding box, numpy array, shape: (N, 1)
    """
    box3d_roi_inds = in_hull(point_cloud[:,0:3], bouding_box_3D)
    return box3d_roi_inds


def get_points_labels(point_cloud, bouding_boxes, bouding_boxes_labels):
    point_cloud_labels = np.zeros((point_cloud.shape[0], 1))
    for box, label in zip(bouding_boxes, bouding_boxes_labels):
        box3d_roi_inds = get_idx_points_in_box3d(point_cloud, box)
        point_cloud_labels[box3d_roi_inds] = label
    return point_cloud_labels

if __name__ == '__main__':
    
    try:
        print("Opening data config file %s" % raw_kitti_config_path)
        data_config = yaml.safe_load(open(raw_kitti_config_path, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()


    dates = [i.name for i in raw_kitti_base_path.glob('*') if i.is_dir()]

    for date in dates:
        date_path = raw_kitti_base_path / date
        drives = [i.name for i in date_path.glob('*') if i.is_dir()]
        for drive in drives:
            drive_number = drive.split('_')[-2]
            data = pykitti.raw(str(raw_kitti_base_path), date, drive_number)
            print(len(data))
            tracklet_labels_file_path = raw_kitti_base_path/ date / drive  / "tracklet_labels.xml"

            output_path = raw_kitti_base_path/ date / drive / "BoudingBoxLabels"
            output_path.mkdir(parents=True, exist_ok=True)
            bouding_boxes, bounding_boxes_labels = load_tracklet(tracklet_labels_file_path)
            for frame_idx in tqdm(range(len(data)), f"{drive}"):
                point_cloud = data.get_velo(frame_idx)
                frame_bouding_boxes = bouding_boxes[frame_idx]
                frame_bouding_boxes_labels = bounding_boxes_labels[frame_idx]
                frame_bouding_boxes_labels_num = [data_config["labels_inv"][i] for i in frame_bouding_boxes_labels]
                point_cloud_labels = get_points_labels(point_cloud, frame_bouding_boxes, frame_bouding_boxes_labels_num)
                point_cloud_labels = point_cloud_labels.astype(np.int32)
                label_file_name = "{:010d}".format(frame_idx)
                np.save(output_path/label_file_name, point_cloud_labels)
# print(load_tracklet(tracklet_labels_file_path))