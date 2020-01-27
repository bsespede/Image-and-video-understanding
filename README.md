# Potential Topics

* detect computer keyboard key hits from video; this is probably only feasible for a reduced set of allowed keys
    * possible approaches:
		* using space time volumes (Filip Ilic PhD) of the hand
		* directly using image segmentation (detecting shape of the hands) and learning individual key presses from these (stationary) images with NN
    * data source:
		* self made
    * potential issues:
		* distinguish pressing/touching keys
* detecting different routines from platfom/cliff diving videos
    * possible approaches:
        * using space time volumes (Filip Ilic PhD)
    * data source: 
        * videos from red bull cliff diving world series competitions
    * potential issues:
        * ?
* cloud forecast based on satellite image sequence or rain forecast based on weather radar image sequence
    * possible approach:
        * R(C)NN
    * data source:
        * one of the datasets from kaggle
        * ZAMG, or equivalent ...
* If we want to keep the effort low, we could do something similar to the examples in Chapters 11-14 from this book: Deep Learning Cookbook, Douwe Osinga

# Space-Time Volume Gesture Classification

## Image Preprocessing

* morphological operations if necessary

## Feature Extraction

Maybe do a principal component analysis on the potential features to find most descriptive features.

1. extract contours
    * perform pointPolygonTest to detect overlapping/occluding volumes
    * contour tracking? from lecture slides: Kass, Witkin, Terzopoulos, 1st ICCV, London, 1987
2. fit shape
    * fitEllipse
    * fitLine
    * minAreaRect
3. compute scalar feature values
	* for 2-3 largest shapes:
        * absolute orientation angle of two biggest objects
        * relative orientation angle
        * relative distance
        * distance to center of bounding box
        * relative size w.r.t. bounding box
        * elongation/compactness
	* number of shapes in bounding box
		* with relative size/area w.r.t. bounding box greater than some threshold

Feature normalization/centering: could become interesting with the absolute angle as 0=pi=2pi -> maybe rough discretization makes sense or (co)sine

## Feature/Token Selection - Outlier Suppression

* minimum relative area of fitted shape w.r.t. boundingbox
* maybe ransac

## Feature Filtering

* matchShapes for STV contours over time: 
    * if there are discontinuities in the matching value -> interpolate
    * or: use it for RANSAC
* median filtering for scalar feature values

## ML

* CNN with 1D convolution layers and pooling layers
* fully connected NN on entire feature-over-time sequences
* RNN ?
* Cost function: cross entropy

# Code structure #

* Data preprocessing:
  * `classifying-tool.py`
  * `visualization-tool.py`
* Feature extraction:
  * `batch_extract_features.py`
* Training:
  * `svm_train_test.py`
* Evaluation/visualisation:
  * `test.py`: run feature extraction on individual video
  * `evaluate.py`: visualize feature extraction & classification result of individual video
  * `box_plot_accuracy.py`
