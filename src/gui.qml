import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Layouts 1.13
import QtMultimedia 5.13

ApplicationWindow {
    id: root
    title: qsTr("Video annotation tool")
    width: 660
    height: 680
    visible: true

    signal setResult(string chosenAction, double actionFirst, double actionSecond, double videoFirst, double videoSecond)
    signal getNextVideo()
    signal getActionList()

    function setNextVideo(videoName) {
        actionCombo.model = ["Choose a label"]
        videoPlayer.source = videoName;
        videoPlayer.pause();
        videoTimer.restart();
    }

    function addToActionList(action) {
        var modelArray = actionCombo.model;
        modelArray.push(action);
        actionCombo.model = modelArray;
    }

    Timer {
        id: videoTimer
        interval: 100
        running: false

        onTriggered: {
            actionCombo.enabled = true
            actionSlider.enabled = true
            videoSlider.enabled = true
            discardBtn.enabled = true
            nextBtn.enabled = false

            actionCombo.currentIndex = 0;
            actionSlider.from = 0;
            actionSlider.first.value = 0;
            actionSlider.to = videoPlayer.duration;
            actionSlider.second.value = videoPlayer.duration;

            videoSlider.from = 0;
            videoSlider.first.value = 0;
            videoSlider.to = videoPlayer.duration;
            videoSlider.second.value = videoPlayer.duration;
        }
    }

    ColumnLayout{
        anchors.margins: 10
        anchors.fill: parent

        Video {
            id: videoPlayer
            width: 640
            height: 480
            fillMode: VideoOutput.Stretch
            autoPlay: false
        }

        RowLayout {

            Text {
                text: "Select label:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            ComboBox {
                id: actionCombo
                Layout.fillWidth: true

                onCurrentIndexChanged: {
                    if (currentIndex != 0) {
                        nextBtn.enabled = true;
                    } else {
                        nextBtn.enabled = false
                    }
                }
            }
        }

        RowLayout {

            Text {
                text: "Label range:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            RangeSlider {
                id: actionSlider
                from: 0
                to: videoPlayer.duration
                first.value: 0
                second.value: actionSlider.to
                Layout.fillWidth: true

                first.onMoved: videoPlayer.seek(first.value)
                second.onMoved: videoPlayer.seek(second.value)
            }
        }

        RowLayout {

            Text {
                text: "Video range:"
                Layout.minimumWidth: 100
                Layout.maximumWidth: 100
            }

            RangeSlider {
                id: videoSlider
                from: 0
                to: videoPlayer.duration
                first.value: 0
                second.value: videoSlider.to
                Layout.fillWidth: true

                first.onMoved: videoPlayer.seek(first.value)
                second.onMoved: videoPlayer.seek(second.value)
            }
        }

        RowLayout {
            spacing: 10

            Button {
                id: discardBtn
                text: "Discard video"
                Layout.fillWidth: true
                onClicked:
                {
                    actionSlider.enabled = false
                    videoSlider.enabled = false
                    discardBtn.enabled = false
                    nextBtn.enabled = false

                    root.getNextVideo()
                }
            }

            Button {
                id: nextBtn
                text: "Next video"
                Layout.fillWidth: true
                enabled: false
                onClicked:
                {
                    var actionFirstPercentage = actionSlider.first.value / videoPlayer.duration
                    var actionSecondPercentage = actionSlider.second.value / videoPlayer.duration
                    var videoFirstPercentage = videoSlider.first.value / videoPlayer.duration
                    var videoSecondPercentage = videoSlider.second.value / videoPlayer.duration
                    var comboText = actionCombo.model[actionCombo.currentIndex]

                    actionCombo.enabled = false
                    actionSlider.enabled = false
                    videoSlider.enabled = false
                    discardBtn.enabled = false
                    nextBtn.enabled = false

                    root.setResult(comboText, actionFirstPercentage, actionSecondPercentage, videoFirstPercentage, videoSecondPercentage)
                    root.getNextVideo()
                    root.getActionList()
                }
            }
        }
    }
}
