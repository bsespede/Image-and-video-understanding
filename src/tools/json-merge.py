import json
import os
import shutil

if __name__ == '__main__':
    results = []
    videoNumber = 0

    scriptDirectory = os.path.dirname(__file__)
    resultsJson = os.path.join(scriptDirectory, "labelling_result.json")
    chosenVideosJsonPath = os.path.join(scriptDirectory, "chosen_videos.json")
    videosSourcePath = os.path.join(scriptDirectory, '../../../data-processed')
    with open(classifiedVideosJsonPath) as classifiedVideosJsonFile:
        classifiedVideosData = json.load(classifiedVideosJsonFile)
        with open(chosenVideosJsonPath) as chosenVideosJsonFile:
            chosenVideosData = json.load(chosenVideosJsonFile)
            for chosenVideos in chosenVideosData:
                for classifiedVideos in classifiedVideosData:
                    if str(classifiedVideos['video_label']) + "_" + classifiedVideos['video_name'] == chosenVideos['video_name']:
                        # merge jsons
                        if chosenVideos['video_status'] == "keep":
                            if videoNumber < 35:
                                results.append({
                                    'video_name': chosenVideos['video_name'],
                                    'video_label': classifiedVideos['video_style'],
                                    'video_start': classifiedVideos['min_action'],
                                    'video_end': classifiedVideos['max_action'],
                                    'video_split': 'test'
                                })
                            else:
                                results.append({
                                    'video_name': chosenVideos['video_name'],
                                    'video_label': classifiedVideos['video_style'],
                                    'video_start': classifiedVideos['min_action'],
                                    'video_end': classifiedVideos['max_action'],
                                    'video_split': 'train'
                                })
                            videoNumber += 1
                        # move files
                        videoSourcePath = os.path.join(videosSourcePath, chosenVideos['video_name'] + ".pkl")
                        videoDestPath = os.path.join(videosSourcePath, chosenVideos['video_status'], chosenVideos['video_name'] + ".pkl")
                        shutil.move(videoSourcePath, videoDestPath)
    #dump results
    with open('videos_final_result.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
