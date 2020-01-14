import numpy as np
import cv2
import pdb
from skimage import img_as_ubyte


class FeatureExtractor:
    """Extract scalar features framewise from STVIs"""

    num_scalar_features_per_stvi = 6

    def __init__(self, stvi_data):
        """:param stvi_data: reference to currently loaded STVI tensor"""

        self.stvi_data = stvi_data
        self.feature_vectors = []
        self.frame_data = []


    def processSTVIs(self, verbose=False, plotting=False):
        print('Processing...')
        num_frames = self.stvi_data.stvis.shape[2]
        self.feature_vectors = np.zeros((num_frames, self.num_scalar_features_per_stvi))
        # for frame_idx in range(num_frames):
        for frame_idx in range(num_frames):
            contour_frame, contours = self.framePreprocessing(self.stvi_data.stvis[:, :, frame_idx], verbose=verbose)
            # plot frame preprocessing results
            if plotting:
                max_contour_frame = np.maximum(contour_frame, self.stvi_data.stvis[:, :, frame_idx, np.newaxis])
                stvi_frame = self.stvi_data.labels_to_falsecolor(self.stvi_data.stvis[:, :, frame_idx])
                self.plotFrame(np.hstack((img_as_ubyte(max_contour_frame), img_as_ubyte(stvi_frame))))

            self.feature_vectors[frame_idx, :] = self.extractScalarFeatures(contours)

        print('done')

    def exportFeatureVector(self, feature_file_path):
        """Save feature vectors to pkl file at given path/filename"""
        ...

    def framePreprocessing(self, stvi_frame, verbose=False):
        stvi_frame_cp = stvi_frame.copy()

        # extract contours
        contours, _ = cv2.findContours(stvi_frame_cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if verbose == True:
            print("Number of extracted contours ", len(contours))
        contour_frame = np.zeros_like(self.stvi_data.labels_to_falsecolor(stvi_frame_cp))

        # find largest area contours
        contour_areas = np.zeros(len(contours))
        for idx, contour in enumerate(contours):
            contour_areas[idx] = cv2.contourArea(contour)

        num_selected_contours = 3
        max_contour_idxs = contour_areas.argsort()[::-1][:num_selected_contours]
        if verbose == True:
            print('max_contour_idxs:', max_contour_idxs)

        for contour_idx in range(len(contours)):
            if contour_idx in max_contour_idxs:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.drawContours(contour_frame, contours, contour_idx, color, thickness=2)

        # find STVI IDs contained in  contours
        max_stvi_id = np.max(stvi_frame_cp.flatten())
        contour_stvi_ids = np.zeros_like(max_contour_idxs)
        stvi_id_histograms = np.zeros((len(max_contour_idxs), max_stvi_id+1))
        for iidx,idx in enumerate(max_contour_idxs):
            filled_contour = np.zeros_like(contour_frame)
            cv2.drawContours(filled_contour, contours, idx, (255, 0, 0), thickness=cv2.FILLED)
            filled_contour_gray = cv2.cvtColor(filled_contour, cv2.COLOR_RGB2GRAY)
            if verbose == True:
                print('nonzero args: ', np.argwhere(filled_contour_gray > 0).shape)
            stvi_ids = np.unique(np.where(filled_contour_gray > 0, stvi_frame_cp, 0))
            if verbose == True:
                print('masked ids: ', stvi_ids, ' all: ', np.unique(stvi_frame_cp))
            stvi_id_histograms[iidx,:] = np.bincount(np.where(filled_contour_gray > 0, stvi_frame_cp, 0).flatten(), minlength=max_stvi_id+1)
            if verbose == True:
                print('hist stvi_ids: ', stvi_id_histograms[iidx,:])

            contour_stvi_ids[iidx] = np.argmax(stvi_id_histograms[iidx,1:]) + 1
            assert(np.isin(stvi_ids, contour_stvi_ids[iidx]).any())

            # contour_cog = np.int0(np.round(np.mean(contours[idx],axis=0)))
            # contour_stvi_ids[iidx] = stvi_frame[contour_cog[0,0], contour_cog[0,1]]
            # contour_stvi_ids[iidx] = stvi_frame[contours[idx][0,0], contours[idx][0,1]]

        if verbose == True:
            print("STVI IDs: ", contour_stvi_ids)

        # detect background contours (TODO?)

        # find IDs matching the largest contour
        num_matching_ids = np.zeros_like(max_contour_idxs)
        reference_ids = np.argwhere(stvi_id_histograms[0,:] > 0)
        for iidx,idx in enumerate(max_contour_idxs):
            num_ids = np.argwhere(stvi_id_histograms[iidx,:]  > 0)
            num_matching_ids[iidx] = len(np.intersect1d(reference_ids, num_ids))

        if verbose == True:
            print("num matching ids: ", num_matching_ids)

        # merge contours based on STVI IDs, relative size (and proximity TODO?)
        min_relative_area = 0.75
        min_matching_ids = 0.66
        reference_contour_area = contour_areas[max_contour_idxs[0]]
        reference_num_ids = len(reference_ids) #sum(stvi_id_histograms[0,:] > 0)
        merged_contour = contours[max_contour_idxs[0]]
        if len(max_contour_idxs) > 1:
            merge_condition = np.logical_and(contour_areas[max_contour_idxs] > (min_relative_area * reference_contour_area), (num_matching_ids / reference_num_ids) >= min_matching_ids)
            if verbose == True:
                print("merge_condition: ", merge_condition, " (", contour_areas[max_contour_idxs] / reference_contour_area, ", ", num_matching_ids / reference_num_ids, ")")
            contour_idxs_to_merge = np.argwhere(merge_condition)
            merged_contour = np.vstack([contours[idx] for idx in max_contour_idxs[contour_idxs_to_merge.flatten()]])

            cv2.drawContours(contour_frame, merged_contour, -1, color=(255, 0, 0), thickness=1)


        # compute bounding box on chosen contour
        bounding_box = cv2.boundingRect(merged_contour)
        cv2.rectangle(contour_frame, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 1)
        

        # fit MBR to contour
        contour_mbrs = []
        contour_mbrs.append(cv2.minAreaRect(merged_contour))
        mbr = cv2.boxPoints(contour_mbrs[-1])
        mbr = np.int0(mbr)
        cv2.drawContours(contour_frame, [mbr], 0, color=(0, 255, 255), thickness=1)

        return contour_frame, merged_contour

    def extractScalarFeatures(self, contour, verbose=False):
        contour_frame = np.zeros(self.stvi_data.stvis.shape[0:2])
        cv2.drawContours(contour_frame, contour, -1, color=(255, 255, 255), thickness=2)
        hu_moments = cv2.HuMoments(cv2.moments(contour_frame)).flatten()
        log_hu_moments = np.sign(hu_moments) * np.log(np.abs(hu_moments))
        if verbose == True:
            print('Hu moments: ', hu_moments)
            print('Hu moments log: ', log_hu_moments)
            self.plotFrame(img_as_ubyte(self.stvi_data.labels_to_falsecolor(contour_frame)))

        return log_hu_moments[0:6]

    def plotFrame(self, frame, window_title="frame"):
        cv2.namedWindow(window_title)
        cv2.imshow(window_title, frame)
        cv2.waitKey(0)
