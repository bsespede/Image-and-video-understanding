import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from skimage import img_as_ubyte

from PySide2.QtCore import QUrl, Slot
from PySide2.QtWidgets import QApplication
from PySide2.QtQml import QQmlApplicationEngine


@Slot(int)
def setVideo(video):
    global _curVideo, _curFrame, _inputs, _window
    print("Video " + str(_curVideo) + "/" + str(len(_inputs)))
    _curVideo = video
    _curFrame = 0
    _window.setVideoSlider()
    updateGUI()


@Slot(int)
def setMode(mode):
    global _curMode
    _curMode = mode
    updateGUI()


@Slot(float)
def setFrame(percentage):
    global _curFrame
    _curFrame = int(percentage * (_inputs[_curVideo]['video_frames'] - 1))
    updateGUI()


@Slot(str)
def setStatus(status):
    global _inputs, _curVideo
    _inputs[_curVideo]['video_status'] = status
    writeJson()


@Slot()
def setNextVideo():
    global _curVideo, _curFrame
    if _curVideo < len(_inputs):
        _curVideo += 1
        _curFrame = 0
        _window.setVideoSlider()
        updateGUI()
    else:
        sys.exit(0)


def initializeGUI():
    global _window, _modes, _inputs
    _modes = {0: 'frames', 1: 'masks', 2: 'opticalflow', 3: 'stvis'}
    for key in _modes.keys():
        _window.addToModesCombo(_modes[key])
    for elem in _inputs:
        _window.addToVideosCombo(elem['video_name'])


def updateGUI():
    global _window, _inputs, _curFrame
    _window.setVideoSource(getCurVideoPath())
    _window.setVideoCombo(_curVideo)


def getCurVideoPath():
    global _inputs, _modes, _curVideo, _curMode
    path = '../../../data-selected'
    name = _inputs[_curVideo]['video_name']
    mode = _modes[_curMode]
    frame = str(_curFrame).zfill(6)
    return path + '/' + name + '/' + mode + '/' + frame + '.jpg'


def readJson():
    global _inputs, _window
    scriptDirectory = os.path.dirname(__file__)
    inputJsonPath = os.path.join(scriptDirectory, "videos_viztool.json")
    with open(inputJsonPath) as jsonFile:
        jsonData = json.load(jsonFile)
        for elem in jsonData:
            _inputs.append({
                'video_name': elem['video_name'],
                'video_status': elem['video_status'],
                'video_frames': elem['video_frames']
            })


def writeJson():
    global _inputs
    with open('videos_viztool.json', 'w') as outfile:
        json.dump(_inputs, outfile, indent=4)


def readPKL(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data


def colorCoding(labels, colorMap):
    total_max = np.max(labels) + 1
    cmap = plt.cm.get_cmap(colorMap, total_max)
    normed = labels / total_max
    retval = cmap(normed)[..., 0:3]
    return img_as_ubyte(retval)


def initJson(videosPath):
    scriptDirectory = os.path.dirname(__file__)
    videosPath = os.path.join(scriptDirectory, videosPath)
    for fileName in os.listdir(videosPath):
        pkl = readPKL(os.path.join(videosPath, fileName))
        name = pkl['Name']
        stvis = colorCoding(pkl['STVIs'], 'viridis')
        opticalflow = colorCoding(np.linalg.norm(pkl['OpticalFlow'], axis=3), 'gray')
        masks = colorCoding(pkl['Masks'], 'binary')
        os.mkdir(name)
        os.mkdir(name + '/stvis')
        os.mkdir(name + '/opticalflow')
        os.mkdir(name + '/masks')
        for frame in range(stvis.shape[2]):
            cv2.imwrite(name + '/stvis/' + str(frame).zfill(6) + '.jpg', stvis[:, :, int(frame), ::-1])
            cv2.imwrite(name + '/opticalflow/' + str(frame).zfill(6) + '.jpg', opticalflow[:, :, int(frame), ::-1])
            cv2.imwrite(name + '/masks/' + str(frame).zfill(6) + '.jpg', masks[:, :, int(frame), ::-1])

        _inputs.append({
            'video_name': name,
            'video_status': 'none',
            'video_frames': int(stvis.shape[2])
        })

    writeJson()


if __name__ == '__main__':
    _curVideo = 0
    _curMode = 0
    _curFrame = 0
    _modes = {}
    _inputs = []
    #initJson('../../../data-processed')
    readJson()

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile('visualization-gui.qml'))

    _window = engine.rootObjects()[0]
    _window.setVideo.connect(setVideo)
    _window.setMode.connect(setMode)
    _window.setFrame.connect(setFrame)
    _window.setStatus.connect(setStatus)
    _window.setNextVideo.connect(setNextVideo)

    initializeGUI()
    updateGUI()

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())
