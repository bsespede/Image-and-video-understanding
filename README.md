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

1. extract contours
    * perform pointPolygonTest to detect overlapping/occluding volumes
2. fit shape
    * fitEllipse
    * fitLine
    * minAreaRect
3. compute scalar feature values:
    * absolute orientation angle of two biggest objects
    * relative orientation angle
    * elongation/compactness

## Feature/Token Selection - Outlier Suppression

* minimum relative area of fitted shape w.r.t. boundingbox
* maybe ransac

## Feature Filtering

* matchShapes for STV contours over time: 
    * if there are discontinuities in the matching value -> interpolate
    * or: use it for RANSAC
* median filtering for scalar feature values

## ML

* CNN with 1D convolution layers
* fully connected NN on entire feature-over-time sequences
* RNN ?
