import numpy as np
import cv2
import matplotlib.pyplot as plt

class CloudGen:
    """
    Cloud Generation class
    """
    def __init__(self, size, x_axis_coeff = 2, y_axis_coeff = 2, cmap= cv2.COLORMAP_OCEAN):
        # image size
        # one size because of picture is square
        self.size = size
        self.max_circle_rad = self.size//5
        self.min_circle_rad = self.size//15
        
        # set colormap
        self.cmap = cmap
        
        # axis for distribution of cloud volume
        # default 50% - on image center
        # self.size//3 - left shift
        self.cloud_center_axis = self.size//x_axis_coeff
        self.cloud_center_y = self.size//y_axis_coeff
        
        # random values border in step_size generation
        self.min_y_step = (self.size//100)*7
        self.max_y_step = (self.size//100)*15
        self.min_x_step = (self.size//100)*7
        self.max_x_step = (self.size//100)*10
        
        # min value for MinMaxScaler
        self.x_min_diff = 1 - ((self.size - self.cloud_center_axis)/self.cloud_center_axis)
        
        # min and max value of light color of cloud
        self.min_light_cloud_color = 120
        self.max_light_cloud_color = 200
        
        # shift value for cloud mask
        self.x_shift = self.size//15
        self.y_shift = self.size//15
        
        # min and max value of dark color of cloud
        self.dark_shifted_min_color = 50
        self.dark_shifted_max_color = 110
        
        # number of attempts to draw pits
        self.pit_count_tries = 10
        
        # max pit shift value
        self.pit_shift_max_value = self.size//3
        
        # pit radius
        self.pit_radius = self.size//10
        
        # pit width
        self.pit_width = self.size//50
        
        
    def gen_cloud(self):
        """
        Returns image with generated cloud
        """
        # gen cloud image and mask for volume
        cloud_mask, mask_color = self._gen_cloud_mask()
        
        # add volume (pits)
        cloud_mask = self._shift_cloud_mask(cloud_mask, mask_color)
        
        # apply colormap
        cloud_img = cv2.applyColorMap(cloud_mask, self.cmap)
        
        # make background color
        cloud_img[cloud_mask==0] = self.gen_random_blue()
        
        return cloud_img
    
    def gen_random_blue(self):
        """
        Returns color in BGR model
        """
        return (np.random.randint(200,255),
                np.random.randint(180,220),
               np.random.randint(130,150))
        
    def _gen_cloud_mask(self):
        """
        Returns mask of cloud
        """
        # start with empty black image
        img = np.zeros((self.size,self.size),np.uint8)
        
        # main color for cloud
        # in RGB model
        cp = np.random.randint(self.min_light_cloud_color,
                               self.max_light_cloud_color)
        color = (cp, cp, cp)
        
        # start coordinates values for circles drawing
        # x = 0
        # y = 50% of image size by default
        x = 0
        y = self.cloud_center_y
        
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
        return img, color
    
    def _shift_cloud_mask(self, mask, color):
        """
        Shift mask of cloud
        """
        # base image for shifted mask
        shifted_mask = np.zeros((self.size,self.size),np.uint8)
        
        # shift mask on defined count of pixels
        # shift_x value on x axis
        # shift_y value on y axis
        shifted_mask[self.y_shift:shifted_mask.shape[0],
                     self.x_shift:shifted_mask.shape[1]] =\
        mask[0:mask.shape[0]-self.y_shift,
             0:shifted_mask.shape[1]-self.x_shift]
        
        # make upper part of difference after shifting white
        shifted_mask[(shifted_mask!=mask)&(shifted_mask==0)] = 255
        
        # make lower part of difference after shifting darker
        shifted_mask[(shifted_mask!=mask)&(mask==0)] = np.random.randint(self.dark_shifted_min_color,
                                                                         self.dark_shifted_max_color)
        
        # draw the pits for cloud volume
        for i in range(self.pit_count_tries):
            
            # preprocessing for pits
            crcl = np.zeros((self.size,self.size),np.uint8)
            ncrcl = np.zeros((self.size,self.size),np.uint8)

            # define "pit" coordinates
            x,y = (self.cloud_center_axis + np.random.randint(self.pit_shift_max_value),
                   self.cloud_center_y + np.random.randint(self.pit_shift_max_value))

            # draw pit
            cv2.circle(crcl, (x,y), self.pit_radius, color, -1)

            # shift pit
            ncrcl[self.pit_width:ncrcl.shape[0],:] = crcl[0:crcl.shape[0]-self.pit_width,:]

            # make shift white
            ncrcl[(crcl!=ncrcl)&(ncrcl==0)] = 255

            # apply shifted "pit" to image
            shifted_mask[(ncrcl!=0)&(mask!=0)] = ncrcl[(ncrcl!=0)&(mask!=0)]

        
        # return shifted mask
        return shifted_mask