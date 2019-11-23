import numpy as np
import cv2
import pdb
from skimage import img_as_ubyte


class FeatureExtractor:
    """Extract scalar features framewise from STVIs"""

    stvi_data = []
    num_scalar_features_per_stvi = 1

    def __init__(self, stvi_data):
        """:param stvi_data: reference to currently loaded STVI tensor"""

        self.stvi_data = stvi_data
        self.feature_vectors = []
        self.frame_data = []


    def processSTVIs(self):
        num_frames = self.stvi_data.stvis.shape[2]
        self.feature_vectors = np.zeros((num_frames, self.num_scalar_features_per_stvi))
        # for frame_idx in range(num_frames):
        for frame_idx in range(num_frames):
            contour_frame, contours = self.framePreprocessing(self.stvi_data.stvis[:, :, frame_idx])
            self.feature_vectors[frame_idx, :] = self.extractScalarFeatures(contours, contour_frame)

    def exportFeatureVector(self, feature_file_path):
        """Save feature vectors to pkl file at given path/filename"""
        ...

    def framePreprocessing(self, stvi_frame):
        stvi_frame_cp = stvi_frame.copy()

        # extract contours
        contours, _ = cv2.findContours(stvi_frame_cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print("Number of extracted contours ", len(contours))
        contour_frame = np.zeros_like(self.stvi_data.labels_to_falsecolor(stvi_frame_cp))

        # find largest area contours
        contour_areas = np.zeros(len(contours))
        for idx, contour in enumerate(contours):
            contour_areas[idx] = cv2.contourArea(contour)

        num_selected_contours = 3
        max_contour_idxs = contour_areas.argsort()[::-1][:num_selected_contours]

        for contour_idx in range(len(contours)):
            if contour_idx in max_contour_idxs:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.drawContours(contour_frame, contours, contour_idx, color, 1)


        # compute bounding box on chosen contours
        bounding_box = cv2.boundingRect(np.vstack([contours[idx] for idx in max_contour_idxs]))
        cv2.rectangle(contour_frame, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 1)
        

        # fit MBR to contours
        contour_mbrs = []
        for contour_idx in max_contour_idxs:
            contour_mbrs.append(cv2.minAreaRect(contours[contour_idx]))
            mbr = cv2.boxPoints(contour_mbrs[-1])
            mbr = np.int0(mbr)
            cv2.drawContours(contour_frame, [mbr], 0, (0, 255, 255), 1)

        return contour_frame, contours

    def extractScalarFeatures(self, contours, contour_frame):
        self.plotFrame(img_as_ubyte(contour_frame))
        return 0

    def plotFrame(self, frame, window_title="frame"):
        cv2.namedWindow(window_title)
        cv2.imshow(window_title, frame)
        cv2.waitKey(0)
