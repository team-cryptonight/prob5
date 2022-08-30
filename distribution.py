import numpy as np
import encrypt
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
def leak(i: int) -> int:
    """
    Hamming weight of i (<= 255)
    """
    # i: 11100000
    i = i - ((i >> 1) & 0x55)  # i: 10010000
    i = (i & 0x33) + ((i >> 2) & 0x33)  # i: 00110000
    i = (i + (i >> 4)) & 0x0F  # i: 00000011
    return i


def leak32(i: int) -> int:
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0F0F0F0F
    i = (i + (i >> 8)) & 0x00FF00FF
    i = (i + (i >> 16)) & 0xFF
    return i


def generate_distribution(func, leakage=leak):
    results = []
    for k in range(256):
        output_leaks = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            o = func(i, k)
            output_leaks[i] = leakage(o)
        results.append(output_leaks)
    return np.vstack(results)


def generate_h_m_h_y(func, leakage=leak):
    K, H = 256, 9
    h_m_h_y = np.zeros((K, H, H), dtype=np.int32)
    leaks = np.array(list(map(leakage, range(256))))  # (256,)
    for k in range(256):
        for i in range(256):
            o = func(i, k)
            h_m_h_y[k, leaks[i], leaks[o]] += 1
    h_m_h_y = h_m_h_y / 256
    return h_m_h_y


def main():
    def func(i, k):
        subPT = np.array([i] * 4, dtype=np.uint8)
        subMK = np.array([k] * 4, dtype=np.uint8)
        o = encrypt.F(subPT, subMK)[2]
        return o
    distribution = generate_distribution(func)
    input_leaks = np.array(list(map(leak, range(256))))
    results = np.zeros((256,), dtype=np.float32)
    for key in range(len(distribution)):
        results[key] = np.corrcoef(np.vstack((input_leaks, distribution[key])))[0, 1]
    plt.plot(results, '.')
    plt.xlabel('key')
    plt.ylabel('correlation of input and output')
    plt.savefig('corr_xor.png')
    plt.cla()

    def func(i, k):
        subPT = np.array([i] * 4, dtype=np.uint8)
        subMK = np.array([k] * 4, dtype=np.uint8)
        o = encrypt.F(subPT, subMK)[3]
        return o
    distribution = generate_distribution(func)
    input_leaks = np.array(list(map(leak, range(256))))
    results = np.zeros((256,), dtype=np.float32)
    for key in range(len(distribution)):
        results[key] = np.corrcoef(np.vstack((input_leaks, distribution[key])))[0, 1]
    plt.plot(results, '.')
    plt.xlabel('key')
    plt.ylabel('correlation of input and output')
    plt.savefig('corr_complex.png')


if __name__=="__main__":
    main()
