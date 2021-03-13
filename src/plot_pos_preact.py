import matplotlib.pyplot as plt

if __name__ == '__main__':

    # fname = "preact.txt"
    fname = "data/ppo_peg_in_preact/ppo_peg_in_preact_s0/progress.txt"
    AverageEpRet1 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet1.append(float(line.split('\t')[1]))

    # fname = "force.txt"
    fname = "data/ppo_peg_in_pos/ppo_peg_in_pos_s0/progress.txt"
    AverageEpRet2 = []
    with open(fname , "r") as f:
        f.readline()
        for line in f.readlines():
            AverageEpRet2.append(float(line.split('\t')[1]))

    # fname = "pos.txt"
    # AverageEpRet3 = []
    # with open(fname , "r") as f:
    #     f.readline()
    #     for line in f.readlines():
    #         AverageEpRet3.append(float(line.split('\t')[1]))

    plt.title('Peg Insert -- compare observation', size=14)
    plt.xlabel('epochs', size=12)
    plt.ylabel('AverageEpRet', size=12)
    # plt.plot(range(len(AverageEpRet1)), AverageEpRet1, color='b', marker='.', label='pre_act') 
    # plt.plot(range(len(AverageEpRet2)), AverageEpRet2, color='r', marker='.', label='pre_act+force')
    # plt.plot(range(len(AverageEpRet3)), AverageEpRet3, color='g', marker='.', label='pre_act+force+pos')
    plt.plot(range(len(AverageEpRet1)), AverageEpRet1, color='b', marker='.', label='preact') 
    plt.plot(range(len(AverageEpRet2)), AverageEpRet2, color='r', marker='.', label='pos') 
    plt.legend(loc='upper left')
    plt.show()