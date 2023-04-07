import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import pandas as pd
import os

def difference(current_frame, previous_frame):
    assert len(current_frame) == len(previous_frame)
    difference = int(sum(abs(current_frame - previous_frame)))
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
    '''
    KunFu Hustle total spent 75.55 seconds.
    Spider Man total spent 91.68 seconds.
    '''
    VIDEO_NAME = 'KungFuHustle'
    SHOW = False
    SAVE_IMG = False

    if VIDEO_NAME == 'KungFuHustle':
        THRESHOLD = 300000
    elif VIDEO_NAME == 'SpiderMan':
        THRESHOLD = 320000

    i = 0
    j = 1
    diffs = []
    shots = []
    frames = []
    times = []

    # Specify the video file path or camera index
    video_source = f"/home/nick/Documents/code/pattern_recognition/HW1/Video/{VIDEO_NAME}_640x480.mp4" 

    # Output file path
    output_dir = "/home/nick/Documents/code/pattern_recognition/HW1/outputs/Original" 
    
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

    # Check if the folder to save image is created successfulliy
    folder_dir = f'images/{VIDEO_NAME}'
    path1 = os.path.join(output_dir, folder_dir)
    isExist = os.path.exists(path1)
    if not isExist:
        os.makedirs(path1)
        os.makedirs(f"{path1}/shot")
        print(f'"{VIDEO_NAME}" folder is created!')
    else:
        print(f'Image Folder "{VIDEO_NAME}" is already exist!')

    # Loop through the frames of the video
    start = time.time()
    while True:
        # Read the next frame
        ret, current_rgb_frame = cap.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        '''
        Implement your video shot detection algorithm here.
        '''
        current_gray_frame = cv2.cvtColor(current_rgb_frame, cv2.COLOR_BGR2GRAY)
        current_frame_histogram = cv2.calcHist([current_gray_frame], [0], None, [256], [0, 256])
        
        if SHOW:
            cv2.imshow('frame', current_gray_frame)
        
        if i == 0:
            previous_gray_frame = current_gray_frame
            previous_rgb_frame = current_rgb_frame
            previous_frame_histogram = current_frame_histogram
        else:
            diff = difference(current_frame_histogram, previous_frame_histogram)
            diffs.append(diff)
            
            if diff > THRESHOLD:
                frames.append(i)
                shots.append(diff)
                txt = find_frame(i, fps)
                times.append(txt)
                if SAVE_IMG:
                    img_dir = f'images/{VIDEO_NAME}/shot_{j}'
                    path2 = os.path.join(output_dir, img_dir)
                    isExist = os.path.exists(path2)
                    if not isExist:
                        os.mkdir(path2)
                    cv2.imwrite(f'{path2}/rgb_{i}.png', current_rgb_frame)
                    cv2.imwrite(f'{path2}/rgb_{i-1}.png', previous_rgb_frame)
                    cv2.imwrite(f'{path2}/gray_{i}.png', current_gray_frame)
                    cv2.imwrite(f'{path2}/gray_{i-1}.png', previous_gray_frame)

                    groups = 256
                    index = np.arange(0, groups)
                    
                    bar_width = 0.4
                    current_y = current_frame_histogram.flatten()
                    previous_y = previous_frame_histogram.flatten()

                    grid = gridspec.GridSpec(2, 4)

                    plt.figure(figsize=(16, 8))
                    plt.title(f"{i} Frame")
                    plt.subplot(grid[0, 0])
                    plt.title(f'Current RGB Frame {i}')
                    plt.imshow(cv2.cvtColor(current_rgb_frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.subplot(grid[0, 1])
                    plt.title(f'Current Gray Frame {i}')
                    plt.imshow(current_gray_frame, cmap='gray')
                    plt.axis('off')
                    plt.subplot(grid[0, 2])
                    plt.title(f'Previous RGB Frame {i-1}')
                    plt.imshow(cv2.cvtColor(previous_rgb_frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.subplot(grid[0, 3])
                    plt.title(f'Previous Gray Frame {i-1}')
                    plt.imshow(previous_gray_frame, cmap='gray')
                    plt.axis('off')
                    plt.subplot(grid[1, :])
                    plt.ylim(0, 80000)
                    p1 = plt.bar(index, current_y, width=bar_width, label='current')
                    p2 = plt.bar(index+bar_width, previous_y, width=bar_width, label='previous')
                    # plt.bar_label(p1)
                    # plt.bar_label(p2)
                    # plt.xticks(index+bar_width/2, ('0-black', '255-white'), fontsize=15)
                    plt.ylabel('Amounts',{'fontsize':15})
                    plt.xlabel('Gray Level',{'fontsize':15})
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{path1}/shot/shot_{j}.png')
                    j += 1

                if SHOW:
                    print(f"frame {i+1}")
                    print(txt)
                    print(f"diff: {diff}")
                    print("----------------")
                    plt.show()
                plt.close()

            previous_frame_histogram = current_frame_histogram
            previous_gray_frame = current_gray_frame
            previous_rgb_frame = current_rgb_frame
                
            del current_frame_histogram
            del current_gray_frame, current_rgb_frame
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
    print(f"Got {len(shots)} shots.")
    end = time.time()
    print(f"Total spent {end-start} seconds.")

if __name__ == "__main__":
    main()