import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_ubyte



class DataProcessor:
    """STV data loading and preprocessing

    Loads the Space-Time Volume of interest data resulting from Filip Ilic
    """


    def __init__(self, stv_path):
        """ :param stvPath: base path to STV data"""

        self.data = []
        self.current_video_name = ""
        self.name = []
        self.images = []
        self.optflow = []
        self.optflow_mag = []
        self.masks = []
        self.bounding_boxes = []
        self.stvis = []

        if os.path.isdir(stv_path):
            self.stv_path = stv_path
        else:
            raise RuntimeError('Invalid path ' + stv_path)

    def processVideo(self, video_name):
        """Load preprocess given video"""

        full_path = self.stv_path + '/' + video_name
        if os.path.isfile(full_path):
            self.current_video_name = full_path
        else:
            raise RuntimeError('Invalid path ' + full_path)

        with open(self.current_video_name, 'rb') as handle:
            self.data = pickle.load(handle)
            self.name = self.data['Name']
            self.images = self.data['Images']
            # self.optflow = self.data['OpticalFlow']
            # self.optflow_mag = np.linalg.norm(self.optflow, axis=3)
            # self.masks = self.data['Masks']
            # self.bounding_boxes = self.data['BoundingBoxes']
            self.stvis = self.data['STVIs']

    def plot_slider_sequence(self, sequence, window_title, frame_number):
        def on_trackbar(val):
            img=cv2.rectangle(sequence[:,:,int(val),::-1], (self.bounding_boxes[int(val)][0:2]), (self.bounding_boxes[int(val)][2:4]), (0,0,255), thickness=3)
            cv2.imshow(window_title, img)

        cv2.namedWindow(window_title)
        cv2.createTrackbar("Frames", window_title, frame_number, sequence.shape[2], on_trackbar)
        on_trackbar(frame_number)
        cv2.waitKey()

    def labels_to_falsecolor(self, labels, cmap_name='viridis'):
        total_max = np.max(labels) + 1
        cmap = plt.cm.get_cmap(cmap_name, total_max)

        normed = labels / total_max
        retval = cmap(normed)[..., 0:3]
        return img_as_ubyte(retval)

    def showFrames(self, frame_number):
        """Shows RGB frames selectable by a slider bar"""

        self.plot_slider_sequence(self.images, self.name + " RGB " + "(" + self.current_video_name + ")", frame_number)

    def showOpticalFlow(self, frame_number):
        """Shows optical flow magnitude frames selectable by a slider bar"""

        self.plot_slider_sequence(self.labels_to_falsecolor(self.optflow_mag, 'binary'), self.name + " Optical Flow Magnitude " + "(" + self.current_video_name + ")", frame_number)

    def showSTVI(self, frame_number):
        """Shows STVI frames selectable by a slider bar"""

        self.plot_slider_sequence(self.labels_to_falsecolor(self.stvis, 'viridis'), self.name + " STVI " + "(" + self.current_video_name + ")", frame_number)

    def showMasks(self, frame_number):
        """Shows mask frames selectable by a slider bar"""

        self.plot_slider_sequence(self.labels_to_falsecolor(self.masks, 'gray'), self.name + " Masks " + "(" + self.current_video_name + ")", frame_number)

