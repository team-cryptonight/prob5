from util import read
from encrypt import generate_subF, test_key, to_str
from distribution import generate_h_m_h_y
from mle import m_y_mle
import matplotlib.pyplot as plt
import numpy as np

def peak_ids(sig, thres=0.06):
    std = np.std(sig, ddof=1, axis=0)
    max_idx = len(std[std > thres])
    std_index = np.argsort(std)[::-1]
    return sorted(std_index[:max_idx])


def analyze_MC_F(sig_MC, sig_F):
    peaks_MC = peak_ids(sig_MC, 0.065)[::2]
    peaks_F = peak_ids(sig_F, 0.068)[::2]
    all_keys = np.zeros((16,), dtype=np.uint8)
    plt.figure(figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            subF = generate_subF(j)
            keys, probs = m_y_mle(
                sig_MC[:, peaks_MC[4 * i + j]],
                sig_F[:, peaks_F[4 * i + j]],
                generate_h_m_h_y(subF)
            )
            all_keys[4 * i + j] = keys[0]
            plt.subplot(4, 4, 4 * i + j + 1)
            if j == 3:
                plt.plot(probs[:50], '.')
            else:
                plt.plot(probs, '.')

    plt.show()
    MK = all_keys
    print("mk", to_str(MK))
    print("is_correct:", all(test_key(MK)))


def main():
    traces = read()
    sig_MC = traces[:, 1000:1500]
    sig_F = traces[:, 1600:2340]
    analyze_MC_F(sig_MC, sig_F)


if __name__=="__main__":
    main()