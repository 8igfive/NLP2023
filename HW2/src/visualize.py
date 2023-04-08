import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def visualize(pi: float, mu0: float, mu1: float, sigma2_0: float, sigma2_1: float):
    df = pd.read_csv('height_data.csv')
    heights = df['height'].values
    
    plt.hist(heights, bins=20, density=True, alpha=0.5)

    # 绘制正态分布曲线1
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mu0, np.sqrt(sigma2_0))
    plt.plot(x, y)

    # 绘制正态分布曲线2
    y2 = norm.pdf(x, mu1, np.sqrt(sigma2_1))
    plt.plot(x, y2)

    # 绘制混合高斯分布曲线
    y3 = pi * norm.pdf(x, mu0, np.sqrt(sigma2_0)) + (1 - pi) * norm.pdf(x, mu1, np.sqrt(sigma2_1))
    plt.plot(x, y3)

    # 添加图例
    plt.legend(['Male Gaussian', 'Female Gaussian', 'Mixture Gaussian'])

    # 显示图像
    plt.show()

if __name__ == '__main__':
    visualize(0.75, 176.26, 164.93, 23.55, 8.63)