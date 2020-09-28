import numpy as np
import cv2
import sys
sys.path.append('../../../../../train')
print(sys.path)
from common.laserscan import SemLaserScan


class PrepareScan(SemLaserScan):
  """Class that contains PointCloudScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']
  EXTENSIONS_LABEL = ['.label', '.npy', '.png']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, calib_params=None, image_width=1241, image_height=376, min_w_angle_degree=0, max_w_angle_degree=360, normalize_rgb=1, normalize_image=1):
    super(PrepareScan, self).__init__(sem_color_dict=None, project=project, H=H, W=W, fov_up=fov_up, fov_down=fov_down, min_w_angle_degree=min_w_angle_degree, max_w_angle_degree=max_w_angle_degree, use_rgb=False)
    self.calib_params = calib_params
    self.image_width = image_width
    self.image_height = image_height
    self.label = None
    self.normalize_rgb = normalize_rgb
    self.normalize_image = normalize_image
    self.reset()

  def open_scan(self, filename, imagename):
    """ Open raw scan and fill in attributes
    """
    points, remissions, _ =  self._open_scan(filename)
    self.filename = filename
    self.imagename = imagename

    #image = Image.open(imagename)
    image = cv2.imread(imagename)
    self.image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    self.set_points(points, remissions, self.image)

  def set_points(self, points, remissions=None, image=None):
    """ Set scan attributes (instead of opening from file)
    """
    if image is not None:
      self.image = image

      if self.normalize_image:
        self.image = self.image / 255.0

    self._set_points(points=points, remissions=remissions, rgb=None)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def filter_points(self,points: np.array, image_width, image_height) -> np.array:
    """
    function finds indexes of points that are within image frame ( within image width and height )
    searches for
      points with x coordinate greater than zero, less than image_width
      points with y coordinate greater than zero, less than image_height
    Args:
      points: points to be filter, shape: number_points,2
      image_width: width of image frame
      image_height: height of image frame
    return:
      indexes of points that satisfy both conditions
    """
    # points with x coordinate greater than zero, less than image_width
    in_w = np.logical_and(points[:, 0] > 0, points[:, 0] < image_width)
    # points with y coordinate greater than zero, less than image_height
    in_h = np.logical_and(points[:, 1] > 0, points[:, 1] < image_height)
    return np.logical_and(in_w, in_h)

  def get_rgb(self,image_coordinates: np.array, image: np.array) -> np.array:
    """
    function gets rgb value from image

    Args:  
      images coordinates to get rgb values of
      image: rgb image
    Returns
      np array with rgb value for every point
    """

    result = np.zeros((image_coordinates.shape[0], 3), dtype=np.float32) # CHANGED FROM float64
    
    for idx in range(image_coordinates.shape[0]):
      # get pixel coordinates of point
      x, y = np.floor(image_coordinates[idx, :]).astype(np.int64)
        
      # set rgb value according to image
      # result[idx, :] = np.array(image.getpixel((int(x),int(y)))).astype(np.float32) # CHANGED FROM float64
      result[idx, :] = image[y, x].astype(np.float32) 
    return result

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
      Function takes no arguments because it can be also called externally
      if the value of the constructor was not set (in case you change your
      mind about wanting the projection)
    """
  
    velo_points_plane_ind = self.points[:, 0] > 0
    
    self.homogenous_points = np.concatenate(
      [self.points, np.ones((self.points.shape[0], 1))], axis=1)
    transformed_velo_to_image = np.matmul(
      self.calib_params.P_rect_20, self.calib_params.T_cam2_velo)
    transformed_velo_to_image = np.dot(
      transformed_velo_to_image, self.homogenous_points.T)
    transformed_velo_to_image = transformed_velo_to_image[:2,:] / transformed_velo_to_image[2, :]

    inds = self.filter_points(transformed_velo_to_image.T,
              image_width=self.image_width, image_height=self.image_height)

    self.image_filter_condition = np.logical_and(velo_points_plane_ind, inds)

    self.points = self.points[self.image_filter_condition,:]
    self.remissions = self.remissions[self.image_filter_condition]

    # get rgb values
    rgb = self.get_rgb(transformed_velo_to_image.T[self.image_filter_condition, :], self.image)
    
    if self.normalize_rgb:
        rgb = rgb / 255.0
    
    self.rgb = rgb

    transformed_velo_to_image = transformed_velo_to_image.T[self.image_filter_condition, :]

    super().do_range_projection()

    rgb = rgb[self.order]
    self.proj_rgb[self.proj_y, self.proj_x] = rgb

  def open_label(self, filename, filter=True):
    """ Open raw scan and fill in attributes
    """
    label = self._open_label(filename=filename)

    if(filter) and np.any(self.image_filter_condition):
      label = label[self.image_filter_condition]
    
    self.label = self.set_label(label)
    
    return self.label