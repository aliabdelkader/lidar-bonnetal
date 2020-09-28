#!/usr/bin/env python3
import numpy as np
from PIL import Image
import math
import argparse
import yaml
import os
from pykitti import odometry
import re
import cv2
import torch
from tqdm import tqdm
import shutil
from PrepareScan import PrepareScan
MAX_POINTS = 150000

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_IMAGE = ['.png']
EXTENSIONS_LABEL = ['.label', '.npy', '.png']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


if __name__== "__main__":
    parser = argparse.ArgumentParser("./prepare_dataset.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='../../config/labels/semantic-kitti.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    parser.add_argument(
        '--save_filtered_scan',
        type=int,
        required=False,
        default=False,
        help='should save scan after filtering by image fov',
    )
    parser.add_argument(
        '--save_filtered_dir',
        type=str,
        required=False,
        default=".",
        help='directory to save filtered scan in',
    )
    parser.add_argument(
        '--include_sets',
        type=str,
        required=True,
        default="train",
        help='which sets to include in pgm stats calculation, train,valid,test',
    )
    parser.add_argument(
        '--min_w_angle',
        type=str,
        required=True,
        default="-40",
        help='min yaw angle in degrees in LiDAR FOV to consider while constructing pgm',
    )
    parser.add_argument(
        '--max_w_angle',
        type=str,
        required=True,
        default="40",
        help='max yaw angle in degrees in LiDAR FOV to consider while constructing pgm',
    )
    parser.add_argument(
        '--normalize_rgb',
        type=int,
        required=False,
        default=False,
        help='normalize rgb values of scan',
    )
    parser.add_argument(
        '--normalize_image',
        type=int,
        required=False,
        default=False,
        help='normalize camera image',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.load(open(FLAGS.data_cfg, 'r'))
        
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    root = os.path.join(FLAGS.dataset, "sequences")
    print(FLAGS)
    save_root = None
    if FLAGS.save_filtered_dir and FLAGS.save_filtered_scan:
      save_root = os.path.join(FLAGS.save_filtered_dir, "sequences")
      if not os.path.isdir(save_root):
        os.makedirs(save_root, exist_ok=True)


    train_sequences=DATA["split"]["train"]
    valid_sequences=DATA["split"]["valid"]
    test_sequences=DATA["split"]["test"]
    sequences = train_sequences.copy()
    sequences.extend(valid_sequences.copy())
    sequences.extend(test_sequences.copy())
    
    include_sets = FLAGS.include_sets.split(",")
    
    include_seq_in_stats = []
    if "train" in include_sets:
      include_seq_in_stats.extend(train_sequences.copy())
    if "valid" in include_sets:
      include_seq_in_stats.extend(valid_sequences.copy())
    if "test" in include_sets:
      include_seq_in_stats.extend(test_sequences.copy())
    print(f"seq included {include_seq_in_stats}")
    sensor = ARCH["dataset"]["sensor"]
    sensor_img_H = sensor["img_prop"]["height"]
    sensor_img_W = sensor["img_prop"]["width"]
    sensor_fov_up = sensor["fov_up"]
    sensor_fov_down = sensor["fov_down"]
    sequence_regex = re.compile('sequences\/(\d+)\/')

    total_scan_files = []
    total_image_files = []

    all_pgms = []

    # fill in with names, checking that all sequences are complete
    for seq in sequences:

        # placeholder for filenames
        scan_files = []
        image_files = []

        # to string
        seq = '{0:02d}'.format(int(seq))

        print("parsing seq {}".format(seq))

        odometry_manager = odometry(root.replace('/sequences', ''), seq)

        # get paths for each
        scan_path = os.path.join(root, seq, "velodyne")
        image_path = os.path.join(root, seq, "image_2")
        label_path = os.path.join(root, seq, "labels")
        if FLAGS.save_filtered_scan:
          save_scan_path = os.path.join(save_root, seq, "velodyne")
          save_image_path = os.path.join(save_root, seq, "image_2")
          save_label_path = os.path.join(save_root, seq, "labels")
          if not os.path.isdir(save_scan_path):
            os.makedirs(save_scan_path, exist_ok=True)
          if not os.path.isdir(save_image_path):
            os.makedirs(save_image_path, exist_ok=True)
          if not os.path.isdir(save_label_path):
            os.makedirs(save_label_path, exist_ok=True)
        # get files
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
        image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(image_path)) for f in fn if is_image(f)]
        label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]
        assert(len(scan_files) == len(image_files))

        # extend list
        total_scan_files.extend(scan_files)
        total_image_files.extend(image_files)

        # sort for correspondance
        scan_files.sort()
        image_files.sort()
        label_files.sort()

        def save_data(scan: PrepareScan, scan_file: str, image_file: str, label_file: str = None):
          """
          function saves the scan points after they were filtered by image field of view
          along with the image, filtered labels
          Args:
            scan: PrepareScan object to be save
            scan_file: path to the original scan file before filtering
            image_file: path to the original image file before filtering
            lable_file: path to the original label file
          """
          path_split = lambda p: os.path.normpath(p).split(os.sep)
          save_scan_file = os.path.join(save_scan_path, path_split(scan_file)[-1] )
          save_image_file = os.path.join(save_image_path, path_split(image_file)[-1])

          # x,y,z,remission,range,r,g,b
          save_scan = np.concatenate([scan.points, scan.remissions.reshape(-1,1), scan.depth.reshape(-1,1), scan.rgb], axis=1)
          save_scan = save_scan.astype(np.float32)
          assert save_scan.shape[0] == scan.points.shape[0]

          np.save(save_scan_file.replace(".bin", ""), save_scan)

          if np.any(scan.image):
            cv2.imwrite(save_image_file, scan.image)
            # shutil.copyfile(image_file, save_image_file)

          if scan.label is not None:
            save_label_file = os.path.join(save_label_path, path_split(label_file)[-1])
            save_label = scan.label.astype(np.int32)
            assert save_scan.shape[0] == scan.label.shape[0]
            # save_label.tofile(save_label_file)
            np.save(save_label_file.replace(".label", ""), save_label)

        def get_pgm(scan: PrepareScan):    
          """
          function returns polar grid map from scan object
          Args:
            scan: PrepareScan object
          
          Returns:
            polar grid map, shape: (8, pgm_height, pgm_width), channels: range,x,y,z,remission,r,g,b
          """  
          # get points and labels
          proj_range = torch.from_numpy(scan.proj_range).clone()
          proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
          proj_rgb = torch.from_numpy(scan.proj_rgb).clone()
          
          proj_remission = torch.from_numpy(scan.proj_remission).clone()
          proj_mask = torch.from_numpy(scan.proj_mask)
        
          proj = torch.cat([proj_range.unsqueeze(0).clone(),
                    proj_xyz.clone().permute(2, 0, 1),
                    proj_remission.unsqueeze(0).clone(),
                    proj_rgb.clone().permute(2, 0, 1)])
          proj = proj * proj_mask.float()
          return proj
        
        def get_scan(scan_file: str, image_file: str, label_file: str = None):
          """
          function creates LaserScan object from scan_file
          Args:
            scan_file: path to original scan file
            image_file: path to rgb image file
          
          Returns:
            LaserScan object

          """
          image_size = cv2.imread(image_file).shape
          scan = PrepareScan(
            project=True,
            H=sensor_img_H,
            W=sensor_img_W,
            fov_up=sensor_fov_up,
            fov_down=sensor_fov_down,
            calib_params=odometry_manager.calib,
            image_width=image_size[1],
            image_height=image_size[0],
            min_w_angle_degree=float(FLAGS.min_w_angle),
            max_w_angle_degree=float(FLAGS.max_w_angle),
            normalize_rgb=int(FLAGS.normalize_rgb),
            normalize_image=int(FLAGS.normalize_image))
          scan.open_scan(scan_file, image_file)
          if label_file is not None:
            scan.open_label(label_file)
          return scan
     
      
        if len(label_files) > 0:
          generator = tqdm(list(zip(scan_files, image_files, label_files)), "processing seq {}".format(seq))
        else:
          generator = tqdm(list(zip(scan_files, image_files)), "processing seq {}".format(seq))
        
        for scan_paths in generator:
          scan = get_scan(*scan_paths)  
          if FLAGS.save_filtered_scan:         
            save_data(scan, *scan_paths)

          if int(seq) in include_seq_in_stats:
            proj = get_pgm(scan)
            all_pgms.append(proj)

    
    if len(all_pgms) > 0:
      stacked_pgms = torch.stack(all_pgms)
      print("Stacked Size: ", stacked_pgms.size())

      means = torch.mean(stacked_pgms, dim=[0,2,3]).cpu().detach().numpy()
      stds = torch.std(stacked_pgms, dim=[0,2,3]).cpu().detach().numpy()
      print("Means: ", means)
      print("Stds: ", stds)
      np.savetxt("pgm_means.csv", means, delimiter=",")
      np.savetxt("pgm_stds.csv", stds, delimiter=",")
