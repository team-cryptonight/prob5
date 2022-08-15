import struct
import numpy as np

def read():
    with open("traces-sc128-10000-6100.bin", "rb") as f:
        N, = struct.unpack('<I', f.read(4))
        print(f"N: {N}")
        L, = struct.unpack('<I', f.read(4))
        print(f"L: {L}")
        traces = []
        for n in range(N):
            traces.append(np.array(struct.unpack('<' + 'd' * L, f.read(L * 8)), dtype=np.float64))
        traces = np.array(traces)
        
    return traces