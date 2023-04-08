import pdb
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from typing import List, Dict

def E_step(theta: Dict[str, float], data: pd.DataFrame) -> List[float]:
    gammas = []
    for i in range(len(data)):
        gamma0 = theta['pi'] * norm.pdf(data['height'][i], loc=theta['mu_0'], scale=theta['sigma2_0']**(1/2))
        gamma1 = (1 - theta['pi']) * norm.pdf(data['height'][i], loc=theta['mu_1'], scale=theta['sigma2_1']**(1/2))
        gammas.append(gamma0 / (gamma0 + gamma1))
    return gammas

def M_step(theta: Dict[str, float], gammas: List[float], data: pd.DataFrame):
    new_pi = sum(gammas) / len(gammas)
    new_mu_0 = sum(gammas[i] * data['height'][i] for i in range(len(gammas))) / sum(gammas)
    new_mu_1 = sum((1 - gammas[i]) * data['height'][i] for i in range(len(gammas))) / (len(gammas) - sum(gammas))
    new_sigma2_0 = sum(gammas[i] * (data['height'][i] - theta['mu_0']) ** 2 for i in range(len(gammas))) / sum(gammas)
    new_sigma2_1 = sum((1 - gammas[i]) * (data['height'][i] - theta['mu_1']) ** 2 for i in range(len(data))) / (len(data) - sum(gammas))
    return {
        'pi': new_pi,
        'mu_0': new_mu_0,
        'mu_1': new_mu_1,
        'sigma2_0': new_sigma2_0,
        'sigma2_1': new_sigma2_1
    }

def EM(theta_0: Dict[str, float], data: pd.DataFrame, threshold: float = 1e-5):
    thetas = [theta_0]
    while True:
        pre_theta = thetas[-1]
        gammas = E_step(pre_theta, data)
        new_theta = M_step(pre_theta, gammas, data)
        thetas.append(new_theta)
        if max(abs(new_theta[key] - pre_theta[key]) for key in new_theta) < threshold:
            break
    return thetas

if __name__ == '__main__':
    DATA_PATH = r'height_data.csv'
    data = pd.read_csv(DATA_PATH)

    theta_true = {
        'pi': 0.75,
        'mu_0': 176,
        'mu_1': 165,
        'sigma2_0': 25,
        'sigma2_1': 9
    }

    # 根据官方数据，我国的男女比例以及男女身高的均值和标准差为
    theta_0 = {
        'pi': 0.5169,
        'mu_0': 169.7,
        'mu_1': 158.,
        'sigma2_0': 10,
        'sigma2_1': 10
    }

    # EM算法
    thetas = EM(theta_0, data, threshold=1e-5)
    print(f'iteration num: {len(thetas)}')
    for p, v in thetas[-1].items():
        print(f'{p}: {v:.4f}')

    # 平均相对误差为
    print(f'average relative error: {sum(abs(thetas[-1][key] - theta_true[key]) / theta_true[key] for key in thetas[-1]) / len(thetas[-1])}')

    # 不同的参数初始化
    pis = np.linspace(0.1, 1., 10)
    mus_0 = np.linspace(170., 180., 10)
    mus_1 = np.linspace(155., 165., 10)
    sigma2s_0 = np.linspace(5., 15., 10)
    sigma2s_1 = np.linspace(5., 15., 10)
    init_thetas = [{'pi': pi, 'mu_0': mu_0, 'mu_1': mu_1, 'sigma2_0': sigma2_0, 'sigma2_1': sigma2_1} 
                    for pi, mu_0, mu_1, sigma2_0, sigma2_1 in zip(pis, mus_0, mus_1, sigma2s_0, sigma2s_1)]

    # 从不同的初始值开始 EM 算法
    res = []
    for theta_0 in tqdm(init_thetas):
        thetas = EM(theta_0, data, threshold=1e-5)
        res.append((len(thetas), thetas[-1]))

    # 输出初始化参数与结果
    for i, (num, theta) in enumerate(res):
        print(f'init theta: {init_thetas[i]}')
        print(f'iteration num: {num}')
        for p, v in theta.items():
            print(f'{p}: {v:.4f}')
        print(f'average relative error: {sum(abs(theta[key] - theta_true[key]) / theta_true[key] for key in theta) / len(theta)}')
        print('-' * 20)             