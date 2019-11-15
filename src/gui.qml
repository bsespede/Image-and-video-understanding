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

    signal setResult(double actionFirst, double actionSecond, double videoFirst, double videoSecond)
    signal getNextVideo()

    function setNextVideo(videoName, styleName) {
        flightStyle.text = styleName;
        videoPlayer.source = videoName;
        videoPlayer.pause();
        videoTimer.restart();
    }

    Timer {
        id: videoTimer
        interval: 100
        running: false

        onTriggered: {
            actionSlider.enabled = true
            videoSlider.enabled = true
            discardBtn.enabled = true
            nextBtn.enabled = true

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

        Text {
            id: flightStyle
            text: "Flight style"
            font.weight: Font.Bold
            horizontalAlignment: Text.AlignHCenter
            Layout.fillWidth: true
        }

        Video {
            id: videoPlayer
            width: 640
            height: 480
            fillMode: VideoOutput.Stretch
            autoPlay: false
        }

        RowLayout {

            Text {
                text: "Action range:"
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
                onClicked:
                {
                    var actionFirstPercentage = actionSlider.first.value / videoPlayer.duration
                    var actionSecondPercentage = actionSlider.second.value / videoPlayer.duration
                    var videoFirstPercentage = videoSlider.first.value / videoPlayer.duration
                    var videoSecondPercentage = videoSlider.second.value / videoPlayer.duration

                    actionSlider.enabled = false
                    videoSlider.enabled = false
                    discardBtn.enabled = false
                    nextBtn.enabled = false

                    root.setResult(actionFirstPercentage, actionSecondPercentage, videoFirstPercentage, videoSecondPercentage)
                    root.getNextVideo()
                }
            }
        }
    }
}
