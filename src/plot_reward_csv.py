import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
def smooth(csv_path,weight=0.97):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return data['Step'].values, smoothed, scalar
    # save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    # save.to_csv('smooth_'+csv_path)


if __name__=='__main__':
    plt.xlabel('steps')
    plt.ylabel('Average episode reward')

    steps1, value1, original_value1 = smooth('run-visiual-tag-episode_reward.csv')
    steps2, value2, original_value2 = smooth('run-force-tag-episode_reward.csv')
    steps3, value3, original_value3 = smooth('run-pre-tag-episode_reward.csv')
    steps4, value4, original_value4 = smooth('run-full-tag-episode_reward.csv')

    plt.plot(steps1, original_value1, color='lightcyan')
    plt.plot(steps3, original_value3, color='paleturquoise')
    plt.plot(steps4, original_value4, color='bisque')
    plt.plot(steps2, original_value2, color='mistyrose')

    plt.plot(steps1, value1, color='dodgerblue', label='No haptics')
    plt.plot(steps3, value3, color='blue', label='No haptics & No vision')
    plt.plot(steps4, value4, color='darkorange', label='Full model')
    plt.plot(steps2, value2, color='red', label='No vision')

    plt.legend(loc='upper left')
    plt.show()