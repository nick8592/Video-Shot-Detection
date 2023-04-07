import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd

def difference(frameA, frameB):
    assert len(frameA) == len(frameB)
    difference = int(sum(abs(frameA - frameB)))
    return difference

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
    diffs = ()
    shots = ()
    frames = ()
    times = ()

    # Specify the video file path or camera index
    video_source = f"/home/nick/Documents/code/pattern_recognition/HW1/Video/{VIDEO_NAME}_640x480.mp4" 

    # Output file path
    output_dir = "/home/nick/Documents/code/pattern_recognition/HW1/outputs/rgb" 
    
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
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        current_frame_r_histogram = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
        current_frame_g_histogram = cv2.calcHist([current_frame], [1], None, [256], [0, 256])
        current_frame_b_histogram = cv2.calcHist([current_frame], [2], None, [256], [0, 256])
        
        if i == 0:
            previous_frame = current_frame
            previous_frame_r_histogram = current_frame_r_histogram
            previous_frame_g_histogram = current_frame_g_histogram
            previous_frame_b_histogram = current_frame_b_histogram
        else:
            r_diff = difference(current_frame_r_histogram, previous_frame_r_histogram)
            g_diff = difference(current_frame_g_histogram, previous_frame_g_histogram)
            b_diff = difference(current_frame_b_histogram, previous_frame_b_histogram)
            diff = (0.4)*r_diff + (0.2)*g_diff + (0.4)*b_diff
            diffs = diffs + (diff,)

            previous_frame = current_frame
            previous_frame_r_histogram = current_frame_r_histogram
            previous_frame_g_histogram = current_frame_g_histogram
            previous_frame_b_histogram = current_frame_b_histogram

            if diff > THRESHOLD:
                frames = frames + (i,)
                shots = shots + (diff,)
                txt = find_frame(i, fps)
                times = times + (txt,)

            if diff > THRESHOLD and SHOW:
                print(f"frame {i}")
                print(txt)
                print(f"diff: {diff}")
                print("----------------")
                plt.figure(figsize=(16, 8))
                plt.subplot(241)
                plt.title('Previous Frame')
                plt.imshow(previous_frame)
                plt.subplot(242)
                plt.hist(previous_frame_r_histogram, 256, [0, 256])
                plt.subplot(243)
                plt.hist(previous_frame_g_histogram, 256, [0, 256])
                plt.subplot(244)
                plt.hist(previous_frame_b_histogram, 256, [0, 256])
                plt.subplot(245)
                plt.title('Current Frame')
                plt.imshow(current_frame)
                plt.subplot(246)
                plt.hist(current_frame_r_histogram, 256, [0, 256])
                plt.subplot(247)
                plt.hist(current_frame_g_histogram, 256, [0, 256])
                plt.subplot(248)
                plt.hist(current_frame_b_histogram, 256, [0, 256])
                plt.show()
                
            del current_frame_r_histogram, current_frame_g_histogram, current_frame_b_histogram
            del current_frame
            del diff, r_diff, g_diff, b_diff
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