import numpy as np
import encrypt
import matplotlib.pyplot as plt


def main():
    results = np.zeros(256, dtype=np.float32)
    for k in range(256):
        outputs = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            subPT = np.array([i] * 4, dtype=np.uint8)
            subMK = np.array([k] * 4, dtype=np.uint8)
            o = encrypt.F(subPT, subMK)[2]
            outputs[i] = o
        results[k] = np.corrcoef(np.vstack((np.arange(256), outputs)))[0, 1]
    plt.plot(results, '.')
    plt.xlabel('key')
    plt.ylabel('covariance of input and output')
    # plt.show()
    plt.savefig('cov.png')


if __name__=="__main__":
    main()
