import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy

METHOD = 'AdaptiveThreshold'
VIDEO_NAME = 'KungFuHustle'
PLOT = 'HISTOGRAM'
FIX_YSCALE = False

if METHOD == 'Original':
    if VIDEO_NAME == 'KungFuHustle':
        THRESHOLD = 300000
    elif VIDEO_NAME == 'SpiderMan':
        THRESHOLD = 320000
elif METHOD == 'AdaptiveThreshold':
    if VIDEO_NAME == 'KungFuHustle':
        THRESHOLD = 80000
    elif VIDEO_NAME == 'SpiderMan':
        THRESHOLD = 80000

input_dir = f"/home/nick/Documents/code/pattern_recognition/HW1/outputs/{METHOD}"

diffs = pd.read_csv(f'{input_dir}/{VIDEO_NAME}_diffs.csv')

if METHOD == 'Original':
    shots = copy.deepcopy(diffs)
    for i in range(len(diffs)):
        if shots['Differences'][i] < THRESHOLD:
            shots['Differences'][i] = 0
elif METHOD == 'AdaptiveThreshold':
    shots = copy.deepcopy(diffs)
    for i in range(len(diffs)):
        if shots['Boundaries'][i] == 0:
            shots['Differences'][i] = 0

plt.figure(figsize=(20, 5))

if PLOT == 'HISTOGRAM':
    for i in tqdm(range(len(diffs))):
        plt.vlines(i, 0, diffs['Differences'][i], colors='b')
        plt.vlines(i, 0, shots['Differences'][i], colors='r')
elif PLOT == 'LINE':
    plt.plot(diffs, color = 'b', label='Original')
    plt.plot(shots, color = 'r', label='Boundary')
    plt.legend(loc='right')

if METHOD == 'Original':
    plt.title(f"{VIDEO_NAME} Shot Detection (Original), Threshold: {THRESHOLD}", {'fontsize':20})
elif METHOD == 'AdaptiveThreshold':
    plt.title(f"{VIDEO_NAME} Shot Detection (AdaptiveThreshold), Threshold: {THRESHOLD}", {'fontsize':20})

if FIX_YSCALE:
    if VIDEO_NAME == 'KungFuHustle':
        plt.ylim(0, 530000)
    elif VIDEO_NAME == 'SpiderMan':
        plt.ylim(0, 620000)
plt.xlabel('Frames',{'fontsize':15})
plt.ylabel('Differences',{'fontsize':15})
plt.show()
