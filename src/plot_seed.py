import matplotlib.pyplot as plt
from sklearn import datasets
def smooth(ret, weight=0.9):
    # data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = ret
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed, scalar

if __name__ == '__main__':

    fname = "data/ppo_peg_in_pos_circle_newreward/ppo_peg_in_pos_s0/progress.txt"
    AverageEpRet1 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet1.append(float(line.split('\t')[1]))
    ret1, ret1_smoothed = smooth(AverageEpRet1)

    fname = "data/ppo_peg_in_pos/ppo_peg_in_pos_s1/progress.txt"
    AverageEpRet2 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet2.append(float(line.split('\t')[1]))
    ret2, ret2_smoothed = smooth(AverageEpRet2)

    fname = "data/ppo_peg_in_pos/ppo_peg_in_pos_s2/progress.txt"
    AverageEpRet3 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet3.append(float(line.split('\t')[1]))
    ret3, ret3_smoothed = smooth(AverageEpRet3)

    plt.xlabel('epochs')
    plt.ylabel('Average episode reward')

    plt.plot(range(len(AverageEpRet1)), ret1_smoothed, color='paleturquoise') 
    plt.plot(range(len(AverageEpRet2)), ret2_smoothed, color='mistyrose') 
    plt.plot(range(len(AverageEpRet3)), ret3_smoothed, color='bisque') 

    plt.plot(range(len(AverageEpRet1)), ret1, color='b', label='seed=0') 
    plt.plot(range(len(AverageEpRet2)), ret2, color='r', label='seed=1') 
    plt.plot(range(len(AverageEpRet3)), ret3, color='darkorange', label='seed=2') 


    plt.legend(loc='upper left')
    plt.show()