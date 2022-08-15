import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from scipy.stats import binom, norm

@dataclass
class SigPoint:
    signal: NDArray[np.float32]
    alpha: np.float32
    beta: np.float32
    var_w: np.float32

    def __post_init__(self):
        self.estimated_hamming_weight = (self.signal - self.beta) / self.alpha


def regression(sig, n=8, p=0.5):
    """
    sig: (N, )
    """
    assert n % 2 == 0
    q_h = np.quantile(sig, binom.cdf(n//2+1, n, p) - binom.pmf(n//2+1, n, p) / 2)
    q_l = np.quantile(sig, binom.cdf(n//2-1, n, p) - binom.pmf(n//2-1, n, p) / 2)
    q_m = np.quantile(sig, 0.5)
    alpha = min(q_h - q_m, q_m - q_l)
    beta = np.quantile(sig, 1/2) - alpha * (n // 2)
    var_w = np.var(sig, ddof=1) - (alpha ** 2) * n * p * (1 - p)
    return SigPoint(sig, alpha, beta, var_w)


def h_given_k(h_m, h_y, h_m_h_y, sigma_m, sigma_y):
    """
    h_m: (N,)
    h_y: (N,)
    h_m_h_y: (K, 9[m], 9[y]) joint pmf of h_m*, h_y*
    """
    if len(h_m_h_y.shape) < 3:
        h_m_h_y = h_m_h_y[np.newaxis, ...]
    w_m = norm(0, sigma_m)
    w_y = norm(0, sigma_y)
    # (K, N, 9[m], 9[y])
    m_noise_prob = w_m.pdf(h_m[..., np.newaxis, np.newaxis] - np.arange(9, dtype=np.float32).reshape(-1, 1))
    y_noise_prob = w_y.pdf(h_y[..., np.newaxis, np.newaxis] - np.arange(9, dtype=np.float32))
    noise_prob = m_noise_prob * y_noise_prob  # eq. 3
    return np.sum(noise_prob * h_m_h_y.reshape(-1, 1, 9, 9), axis=(-2, -1))  # eq. 2


def log_k_given_h(h_m, h_y, h_m_h_y, sigma_m, sigma_y):
    """
    h_m: (N,)
    h_y: (N,)
    h_m_h_y: (K, 9[m], 9[y]) joint pmf of h_m*, h_y*
    """
    return np.sum(np.log(h_given_k(h_m, h_y, h_m_h_y, sigma_m, sigma_y)), axis=-1)


def m_y_mle(sig_m, sig_y, h_m_h_y):
    """
    sig_m: (N,)
    sig_y: (N,)
    h_m_h_y: (K, 9[m], 9[y])

    return: (K,)[key], (K,)[prob]
    """
    sig_point_m = regression(sig_m)
    sig_point_y = regression(sig_y)
    prob_k = log_k_given_h(
        sig_point_m.estimated_hamming_weight,
        sig_point_y.estimated_hamming_weight,
        h_m_h_y,
        np.sqrt(sig_point_m.var_w) / sig_point_m.alpha,
        np.sqrt(sig_point_y.var_w) / sig_point_y.alpha,
        )
    return np.argsort(prob_k)[::-1], np.sort(prob_k)[::-1]


def main():
    from util import read
    from encrypt import F
    from distribution import generate_h_m_h_y
    import matplotlib.pyplot as plt
    traces = read()
    sig_m = traces[:, 1022]
    sig_y = traces[:, 1639]
    j = 0
    def func(i, k):
        subPT = np.array([i] * 4, dtype=np.uint8)
        subMK = np.array([k] * 4, dtype=np.uint8)
        o = F(subPT, subMK)[j]
        return o
    h_m_h_y = generate_h_m_h_y(func)
    keys, probs = m_y_mle(sig_m, sig_y, h_m_h_y)
    plt.plot(probs, '.')
    print(keys)
    plt.show()


if __name__=="__main__":
    main()