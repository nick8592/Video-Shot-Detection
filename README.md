# Video Shot Detection
## Rules
 - Programs must be based on C/C++ or Python w/wo window interface
## Requirements
Please complete the following two methods
1. Using **histogram feature** to implement the video shot detection.
2. Design **your own algorithm** to implement the video shot detection.   
Note : Please indicate the frame index where the shot changes.
## Usage
Remember to change
 - `video_source` - your video file path
 - `output_dir` - your output folder path   

Run Original
```
python original.py
```
Run AdaptiveThreshold
```
python adaptivethreshold.py
```

Plot Shot Detection Histogram
Remember to change
 - `input_dir` - your .csv folder path  
```
python histogram.py
```

## Flowchart
![Flowchart](https://github.com/nick8592/Pattern-Recognition-Class/blob/main/Video%20Shot%20Detection/Flowchart.png)
## Shot Detection Histogram
### KungFu Hustle
![Original KungFu Hustle Shot Histogram](https://github.com/nick8592/Pattern-Recognition-Class/blob/main/Video%20Shot%20Detection/outputs/Original/KungFuHustle_ShotDetection.png)
![Adaptive KungFu Hustle Shot Histogram](https://github.com/nick8592/Pattern-Recognition-Class/blob/main/Video%20Shot%20Detection/outputs/AdaptiveThreshold/KungFuHustle_ShotDetection.png)
### Spider Man
![Origianl Spider Man Shot Histogram](https://github.com/nick8592/Pattern-Recognition-Class/blob/main/Video%20Shot%20Detection/outputs/Original/SpiderMan_ShotDetection.png)
![Adaptive Spider Man Shot Histogram](https://github.com/nick8592/Pattern-Recognition-Class/blob/main/Video%20Shot%20Detection/outputs/AdaptiveThreshold/SpiderMan_ShotDetection.png)
