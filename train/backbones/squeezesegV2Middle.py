#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import imp
import __init__ as booger
Fire = imp.load_source("Fire", booger.TRAIN_PATH + '/backbones/squeezesegV2.py').Fire
CAM = imp.load_source("CAM", booger.TRAIN_PATH + '/backbones/squeezesegV2.py').CAM
SqueezesegLidarEncoder = imp.load_source("Backbone", booger.TRAIN_PATH + '/backbones/squeezesegV2.py').Backbone


class SqueezesegImageEncoder(nn.Module):
  """
     Class for Squeezeseg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    # Call the super constructor
    super(SqueezesegImageEncoder, self).__init__()
    self.bn_d = params["bn_d"]
    self.drop_prob = params["dropout"]
    self.OS = params["OS"]
    self.input_depth = 3
    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # encoder
    self.conv1a = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=3,
                                          stride=[1, self.strides[0]],
                                          padding=1),
                                nn.BatchNorm2d(64, momentum=self.bn_d),
                                nn.ReLU(inplace=True),
                                CAM(64, bn_d=self.bn_d))
    self.conv1b = nn.Sequential(nn.Conv2d(self.input_depth, 64, kernel_size=1,
                                          stride=1, padding=0),
                                nn.BatchNorm2d(64, momentum=self.bn_d))
    self.fire23 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[1]],
                                             padding=1),
                                Fire(64, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d),
                                Fire(128, 16, 64, 64, bn_d=self.bn_d),
                                CAM(128, bn_d=self.bn_d))
    self.fire45 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                             stride=[1, self.strides[2]],
                                             padding=1),
                                Fire(128, 32, 128, 128, bn_d=self.bn_d),
                                Fire(256, 32, 128, 128, bn_d=self.bn_d))
    self.fire6789 = nn.Sequential(nn.MaxPool2d(kernel_size=3,
                                               stride=[1, self.strides[3]],
                                               padding=1),
                                  Fire(256, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 48, 192, 192, bn_d=self.bn_d),
                                  Fire(384, 64, 256, 256, bn_d=self.bn_d),
                                  Fire(512, 64, 256, 256, bn_d=self.bn_d))

    # output
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 512

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    os *= 2

    x, skips, os = self.run_layer(x, self.fire23, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.fire45, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.fire6789, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth


class Backbone(nn.Module):
  """
     Class for SqueezesegMiddle. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    # Call the super constructor
    super(Backbone, self).__init__()
    self.bn_d = params["bn_d"]
    self.lidar_encoder = SqueezesegLidarEncoder(params)
    self.image_encoder = SqueezesegImageEncoder(params)

    self.fuse = nn.Sequential(
      nn.Conv2d(self.lidar_encoder.get_last_depth() + self.image_encoder.get_last_depth(), 
                self.lidar_encoder.get_last_depth(),
                kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
      nn.BatchNorm2d(self.lidar_encoder.get_last_depth(), momentum=self.bn_d),
      nn.LeakyReLU(0.1)
    )


  
  def forward(self, lidar, camera):

    image_features, _ = self.image_encoder(camera)
    # filter input
    x, skips = self.lidar_encoder(lidar)
    
    cat = torch.cat((x, image_features), dim=1)
    
    fused = self.fuse(cat)
    
    return fused, skips

  def get_last_depth(self):
    return self.lidar_encoder.get_last_depth()

  def get_input_depth(self):
    return self.lidar_encoder.get_input_depth()