import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == '__main__':

    delta_x = 0.001
    x = np.arange(-0.1, 0.1, delta_x)
    delta_z = 0.001
    z = np.arange(0, 0.24, delta_z)

    X, Z = np.meshgrid(x, z)

    def reward1(_x, _z):
        _reward = np.zeros(_x.shape)
        for i in range(_x.shape[0]):
            for j in range(_x.shape[1]):
                x = _x[i][j]
                z = _z[i][j]
                sz = z - 0.06
                s = math.sqrt(x * x + sz * sz) 
                sxy = abs(x)
                s_max = 0.06
                sxy_max = 0.005
                delta_z = 0.005
                if s > s_max:#靠近阶段
                    reward = 2 - math.tanh(10 * s) - math.tanh(10 * sxy)
                elif sxy > sxy_max or sz > 3 * delta_z:#对齐阶段
                    reward = 2 - 5 * sxy - 5 * sz
                elif z > delta_z:#插入阶段
                    reward = 4 - 2 * (sz / 0.06)
                else:#完成阶段
                    reward = 10
                _reward[i][j] = reward
        return _reward
    
    def reward2(_x, _z):
        _reward = np.zeros(_x.shape)
        for i in range(_x.shape[0]):
            for j in range(_x.shape[1]):
                x = _x[i][j]
                z = _z[i][j]
                sz = z - 0.06
                s = math.sqrt(x * x + sz * sz) 
                sxy = abs(x)
                sxy_max = 0.005
                delta_z = 0.005
                if sxy < sxy_max:   #对齐阶段
                    if sz < 0:    #插入阶段
                        if z < delta_z:   #完成阶段
                            reward = 10
                        else:
                            reward = 4 - 2 * (sz / 0.06)
                    else:
                        reward = 2 - 5 * sxy - 5 * sz
                else:   #靠近阶段
                    reward = 2 - math.tanh(10 * s) - math.tanh(10 * sxy)
                _reward[i][j] = reward
        return _reward

    plt.subplot(121)
    plt.contourf(X, Z, reward1(X, Z))
    plt.contour(X, Z, reward1(X, Z))
    box_x = np.arange(-0.06, 0.07, 0.01)
    box_z = np.zeros(box_x.shape) + 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_z = np.arange(0, 0.06, 0.001)
    box_x = np.zeros(box_z.shape) - 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_x = np.zeros(box_z.shape) + 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_x = np.zeros(box_z.shape) + 0.011
    plt.plot(box_x, box_z, color='Gray')
    box_x = np.zeros(box_z.shape) - 0.011
    plt.plot(box_x, box_z, color='Gray')
    plt.axis("equal")
    plt.title('reward1')
    plt.xlabel('horizontal')
    plt.ylabel('vertical')

    plt.subplot(122)
    plt.contourf(X, Z, reward2(X, Z))
    plt.contour(X, Z, reward2(X, Z))
    box_x = np.arange(-0.06, 0.07, 0.01)
    box_z = np.zeros(box_x.shape) + 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_z = np.arange(0, 0.06, 0.001)
    box_x = np.zeros(box_z.shape) - 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_z = np.arange(0, 0.06, 0.001)
    box_x = np.zeros(box_z.shape) + 0.06
    plt.plot(box_x, box_z, color='Gray')
    box_x = np.zeros(box_z.shape) + 0.011
    plt.plot(box_x, box_z, color='Gray')
    box_x = np.zeros(box_z.shape) - 0.011
    plt.plot(box_x, box_z, color='Gray')
    plt.axis("equal")
    plt.title('reward2')
    plt.xlabel('horizontal')
    plt.ylabel('vertical')
    plt.show()