# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import imp
import __init__ as booger

# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}
BasicBlock = imp.load_source("BasicBlock", booger.TRAIN_PATH + '/backbones/darknet.py').BasicBlock
DarknetLidarEncoder = imp.load_source("Backbone", booger.TRAIN_PATH + '/backbones/darknet.py').Backbone
    
class DarknetImageEncoder(nn.Module):
  """
   ImageEncoder for with darknet as backbone
  """

  def __init__(self, params):
    super(DarknetImageEncoder, self).__init__()
    
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.OS = params["OS"]
    self.layers = params["extra"]["layers"]
    self.input_depth = 3
    print("Using ImageEncoder" + str(self.layers) + " Backbone")

    # stride play
    self.strides = [2, 2, 2, 2, 2]
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

    # check that darknet exists
    assert self.layers in model_blocks.keys()

    # generate layers depending on darknet type
    self.blocks = model_blocks[self.layers]

    # input layer
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    # encoder
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], bn_d=self.bn_d)
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], bn_d=self.bn_d)
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], bn_d=self.bn_d)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[3], bn_d=self.bn_d)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[4], bn_d=self.bn_d)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
    layers = []

    #  downsample
    layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                     kernel_size=3,
                                     stride=[1, stride], dilation=1,
                                     padding=1, bias=False)))
    layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    inplanes = planes[1]
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(inplanes, planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

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

    # first layer
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu1, skips, os)

    # all encoder blocks with intermediate dropouts
    x, skips, os = self.run_layer(x, self.enc1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

class Backbone(nn.Module):
  """
     Class for DarknetMiddle
  """

  def __init__(self, params):
    super(Backbone, self).__init__()
    self.bn_d = params["bn_d"]
    
    self.lidar_encoder = DarknetLidarEncoder(params)
    self.image_encoder = DarknetImageEncoder(params)
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
