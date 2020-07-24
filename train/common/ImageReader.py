import cv2

class ImageReader():

    def __init__(self, resize_height: int, resize_width: int):
        """
        Constructor

        Args:
            resize_height = height to resize images to
            resize_width = width to resize image to
        """
        self.resize_height = resize_height
        self.resize_width = resize_width

        assert isinstance(self.resize_height, int), "error expecting integer for height"
        assert isinstance(self.resize_width, int), "error expecting integer for width"
    
    def open_image(self, filename: str, resize: bool =True):
        """
        get image 

        Args:
            filename: path to image file
            resize: boolean, resize open image or not
        """
        
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        if resize:
            image = cv2.resize(image, (self.resize_width, self.resize_height))
        
        return image
