import matplotlib.pyplot as plt

def smooth(ret, weight=0.8):
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

    # fname = "preact.txt"
    fname = "data/ppo_peg_in_hr_nr/ppo_peg_in_hr_nr_s0/progress.txt"
    AverageEpRet1 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet1.append(float(line.split('\t')[1]))
    ret1, ret1_smoothed = smooth(AverageEpRet1)

    # fname = "force.txt"
    fname = "data/ppo_peg_in_hr_or/ppo_peg_in_hr_or_s0/progress.txt"
    AverageEpRet2 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet2.append(float(line.split('\t')[1]))
    ret2, ret2_smoothed = smooth(AverageEpRet2)
    

    # fname = "pos.txt"
    # AverageEpRet3 = []
    # with open(fname , "r") as f:
    #     f.readline()
    #     for line in f.readlines():
    #         AverageEpRet3.append(float(line.split('\t')[1]))

    # plt.title('Peg Insert -- compare observation', size=14)
    plt.xlabel('epochs')
    plt.ylabel('Average episode reward')
    # plt.plot(range(len(AverageEpRet1)), AverageEpRet1, color='b', marker='.', label='pre_act') 
    # plt.plot(range(len(AverageEpRet2)), AverageEpRet2, color='r', marker='.', label='pre_act+force')
    # plt.plot(range(len(AverageEpRet3)), AverageEpRet3, color='g', marker='.', label='pre_act+force+pos')
    plt.plot(range(len(ret2)), ret2_smoothed, color='dodgerblue') 
    plt.plot(range(len(ret1)), ret1_smoothed, color='salmon')

    # plt.plot(range(len(ret2)), ret2, color='b', label='reward1') 
    # plt.plot(range(len(ret1)), ret1, color='r', label='reward2') 
    plt.legend(loc='upper left')
    plt.show()