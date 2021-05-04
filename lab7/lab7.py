import numpy as np
import math
from scipy.stats import norm, laplace, uniform
import csv


def method_maximum_likelihood(sel):
    return np.mean(sel), np.var(sel)


def get_k(n):
    return math.ceil(1.72 * n ** (1 / 3))


def get_intervals(a, b, k):
    return np.linspace(a, b, num=k-1)


def get_n(sel, k, intervals):
    n = [0 for _ in range(k)]
    for x_i in sel:
        if x_i <= intervals[0]:
            n[0] += 1
        if x_i > intervals[k-2]:
            n[k-1] += 1
        for i in range(k - 2):
            if intervals[i] < x_i <= intervals[i + 1]:
                n[i + 1] += 1
    return n


def get_p(intervals):
    a = [norm.cdf(-math.inf)] + [norm.cdf(a_i) for a_i in intervals] + [norm.cdf(math.inf)]
    p = []
    for i in range(1, len(a)):
        p.append(a[i] - a[i - 1])
    return p


def hypothesis_testing(intervals, k, sel, size):
    n = get_n(sel, k, intervals)
    p = get_p(intervals)
    np = [size * p_i for p_i in p]
    n_np = [n_i - np_i for n_i, np_i in zip(n, np)]
    z = [(n_np_i ** 2) / np_i for n_np_i, np_i in zip(n_np, np)]
    return n, p, np, n_np, z


def build_table(a_1, a_n, k, sel, size, dist):
    intervals = get_intervals(a_1, a_n, k)
    print(intervals)
    n, p, np, n_np, z = hypothesis_testing(intervals, k, sel, size)
    with open(dist + str(size) + '.csv', mode='w', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        file_writer.writerow(('i', 'delta_i', 'n_i', 'p_i', 'np_i', 'n_i - np_i', '(n_i-np_i)^2/np_i'))
        for i in range(k):
            file_writer.writerow((str(i), ' ', str(n[i]), str(p[i]), str(np[i]), str(n_np[i]), str(z[i])))
        file_writer.writerow(('sum', '-', str(sum(n)), str(sum(p)), str(sum(np)), str(sum(n_np)), str(sum(z))))


if __name__ == '__main__':
    mu = 0
    sigma = 1
    alpha = 0.05
    a_n = 1.56

    size = 100
    k = get_k(size)
    chi_2 = 14.07
    a_1 = -1.01

    selection = np.random.normal(mu, sigma, size)
    m_, s_ = method_maximum_likelihood(selection)

    print('Метод максимального правдоподобия')
    print('mu=' + str(m_) + ', sigma=' + str(s_))

    print('Метод хи-квадрат для нормального распределния n = 100')
    print('k = ' + str(k))
    print('alpha = ' + str(alpha))
    print('chi^2 = ' + str(chi_2))

    build_table(a_1, a_n, k, selection, size, 'Norm')

    size = 20
    k = get_k(size)
    chi_2 = 9.49
    a_1 = -1.5

    print('Метод хи-квадрат для распределния Лапласа n = 20')
    print('k = ' + str(k))
    print('alpha = ' + str(alpha))
    print('chi^2 = ' + str(chi_2))

    print('Метод хи-квадрат для равномерного распределния n = 20')
    print('k = ' + str(k))
    print('alpha = ' + str(alpha))
    print('chi^2 = ' + str(chi_2))

    selection_uniform = uniform.rvs(-math.sqrt(3), 2 * math.sqrt(3), size)
    selection_laplace = laplace.rvs(mu, sigma / math.sqrt(2), size)

    build_table(a_1, a_n, k, selection_uniform, size, 'Uniform')
    build_table(a_1, a_n, k, selection_laplace, size, 'Laplace')

