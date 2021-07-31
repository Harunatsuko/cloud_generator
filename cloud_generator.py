import numpy as np
import cv2
import matplotlib.pyplot as plt

class CloudGen:
    """
    Cloud Generation class
    """
    def __init__(self, size):
        # image size
        # one size because of picture is square
        self.size = size
        self.max_circle_rad = self.size//5
        self.min_circle_rad = self.size//15
        
        # axis for distribution of cloud volume
        # default 50% - on image center
        # self.size//3 - left shift
        # self.size//1.5 - right shift
        self.cloud_center_axis = self.size//2
        
        # random values border in step_size generation
        self.min_y_step = (self.size//100)*7
        self.max_y_step = (self.size//100)*15
        self.min_x_step = (self.size//100)*7
        self.max_x_step = (self.size//100)*10
        
        # min value for MinMaxScaler
        self.x_min_diff = 1 - ((self.size - self.cloud_center_axis)/self.cloud_center_axis)
        
    def gen_cloud(self):
        """
        Returns image with generated cloud
        """
        # start with empty black image
        img = np.zeros((self.size,self.size),np.uint8)
        
        # white color for cloud
        # in RGB model
        color = (255 ,255, 255)
        
        # start coordinates values for circles drawing
        # x = 0
        # y = 50% of image size
        # self.size //2 == self.size//100*50 - 50% of image size
        x = 0
        y = self.size //2
        
        # start drawing
        y_step = np.random.randint(self.min_y_step, self.max_y_step)
        x_step = np.random.randint(self.min_x_step, self.max_x_step)
        
        # for oddness check
        i = 0
        # while x is not end of image
        while x < self.size:
            i = i+1
            
            # proximity of the circle center to the axis of the center of the cloud 
            # the closer to the center, the larger the radius of the circle
            diff = 1 - np.abs((x-self.cloud_center_axis)/self.cloud_center_axis)

            # alignment of the value if the axis is off-center 
            # aligning by MinMaxScaling
            if x > self.cloud_center_axis:
                diff = (diff - self.x_min_diff) / (1-self.x_min_diff)
                
            # define circle radius
            # multiply the maximum possible radius
            # by the proximity to the center axis 
            cs = int(np.round(self.max_circle_rad*diff))

            # to not draw really small circles
            if cs > self.min_circle_rad:
                
                # circle drawing
                img = cv2.circle(img, (x,y), cs, color, -1)
                
            # change x coordinate for next circle
            x = x + x_step
            # random change y_step for next circle
            y_step = np.random.randint(self.min_y_step, self.max_y_step)

            # checking the deviation from the horizontal axis
            if i%2 == 0:
                y = y + y_step
            else:
                y = y - y_step
        
        # return image with generated cloud
        return img