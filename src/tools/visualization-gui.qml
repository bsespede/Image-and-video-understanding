import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtMultimedia 5.13

ApplicationWindow {
    id: root
    title: qsTr("Visualization tool")
    width: 660
    height: 680
    visible: true

    signal setVideo(int video)
    signal setMode(int mode)
    signal setFrame(double percentage)
    signal setStatus(string status)
    signal setNextVideo()

    function addToVideosCombo(video) {
        var modelArray = videoCombo.model;
        modelArray.push(video);
        videoCombo.model = modelArray;
    }

    function addToModesCombo(mode) {
        var modelArray = modeCombo.model;
        modelArray.push(mode);
        modeCombo.model = modelArray;
    }

    function setVideoSource(frameSource) {
        frame.source = frameSource;
    }

    function setVideoSlider() {
        frameSlider.value = 0;
    }

    function setVideoCombo(videoIndex) {
        videoCombo.currentIndex = videoIndex
    }

    ColumnLayout{
        anchors.margins: 10
        anchors.fill: parent

        Image {
            id: frame
            width: 640
            height: 480
            fillMode: VideoOutput.Stretch
        }

        RowLayout {

            Text {
                text: "Select video:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            ComboBox {
                id: videoCombo
                Layout.fillWidth: true
                model: []
                onCurrentIndexChanged: {
                    root.setVideo(currentIndex);
                }
            }
        }

        RowLayout {
            Text {
                text: "Select mode:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            ComboBox {
                id: modeCombo
                Layout.fillWidth: true
                model: []
                onCurrentIndexChanged: {
                   root.setMode(currentIndex)
                }
            }
        }

        RowLayout {
            Text {
                text: "Video frame:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            Slider {
                id: frameSlider
                Layout.fillWidth: true
                onMoved: {
                    root.setFrame(value)
                }
            }
        }

        RowLayout {
            spacing: 10

            Button {
                id: discardBtn
                text: "Discard video"
                Layout.fillWidth: true
                onClicked: {
                    root.setStatus('discarded')
                    root.setNextVideo()
                }
            }

            Button {
                id: keepBtn
                text: "Keep video"
                Layout.fillWidth: true
                onClicked: {
                    root.setStatus('keep')
                    root.setNextVideo()
                }
            }
        }
    }
}
