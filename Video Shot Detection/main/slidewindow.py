import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd

def difference(current_frame, previous_frame):
    difference = 0
    for i in range(len(previous_frame)):
        difference += int(sum(abs(current_frame - previous_frame[i])))
    avg_difference = difference / len(previous_frame)
    return avg_difference

def find_frame(frame, fps):
    seconds = frame / fps
    second = round(seconds % 60)
    minute = round(seconds // 60)
    hour = round(minute // 60)
    if minute > 60:
        minute -= 60
    text = f"@{hour}h:{minute}m:{second}s"
    return text

def main():
    VIDEO_NAME = 'KungFuHustle'
    THRESHOLD = 400000
    # VIDEO_NAME = 'SpiderMan'
    # THRESHOLD = 500000
    SHOW = False

    i = 0
    diffs = []
    shots = []
    frames = []
    times = []
    previous_frame_histogram = []

    # Specify the video file path or camera index
    video_source = f"/home/nick/Documents/code/pattern_recognition/HW1/Video/{VIDEO_NAME}_640x480.mp4" 

    # Output file path
    output_dir = "/home/nick/Documents/code/pattern_recognition/HW1/outputs/slidewindow" 
    
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_source)

    # Check if the video source is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return
    else:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Videp Name: {VIDEO_NAME}")
        print(f"Total has {length} frames.")
        print(f"Video use {fps} fps.")
        print(f"(width, height) = ({width}, {height})")

    # Loop through the frames of the video
    start = time.time()
    while True:
        # Read the next frame
        ret, current_frame = cap.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        '''
        Implement your video shot detection algorithm here.
        '''
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        current_frame_histogram = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
        
        if i == 0:
            previous_frame = current_frame
            previous_frame_histogram.append(current_frame_histogram)
        elif i < 48:
            previous_frame_histogram.append(current_frame_histogram)
        else:
            diff = difference(current_frame_histogram, previous_frame_histogram)
            diffs.append(diff)
            previous_frame_histogram.pop(0)
            previous_frame_histogram.append(current_frame_histogram)

            if diff > THRESHOLD:
                frames.append(i)
                shots.append(diff)
                txt = find_frame(i, fps)
                times.append(txt)

            if diff > THRESHOLD and SHOW:
                print(f"frame {i}")
                print(txt)
                print(f"diff: {diff}")
                print("----------------")
                plt.figure(figsize=(10, 8))
                plt.subplot(221)
                plt.title('Previous Frame')
                plt.imshow(previous_frame, cmap='gray')
                plt.subplot(222)
                plt.hist(previous_frame_histogram[-1], 256, [0, 256])
                plt.subplot(223)
                plt.title('Current Frame')
                plt.imshow(current_frame, cmap='gray')
                plt.subplot(224)
                plt.hist(current_frame_histogram, 256, [0, 256])
                plt.show()
            
            previous_frame = current_frame
                
            del current_frame_histogram
            del current_frame
            del diff
            del ret

        if i % 10000 == 0:
            print(f"frame {i}")
        i += 1

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    dict = {'Frame': frames, 'Differences': shots, 'Time': times} 
    df = pd.DataFrame(dict) 
    df.to_csv(f'{output_dir}/{VIDEO_NAME}_shots.csv', index=False)

    dict = {'Differences': diffs} 
    df = pd.DataFrame(dict) 
    df.to_csv(f'{output_dir}/{VIDEO_NAME}_diffs.csv', index=False)

    print(f"Processed {i} frames.")
    end = time.time()
    print(f"Total spent {end-start} seconds.")

if __name__ == "__main__":
    main()