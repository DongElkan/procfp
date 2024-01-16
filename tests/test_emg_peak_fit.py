import numpy as np
import matplotlib.pyplot as plt

from procfp.core import construct_peak, fit_peak


def test_peak_fit():
    peaks = np.genfromtxt(r"peak.txt", dtype=np.float64)
    peaks -= peaks.min()
    peaks += 0.00000001
    rt = np.arange(peaks.shape[0], dtype=np.float64) * 0.4 + 0.1

    param = fit_peak(rt, peaks)
    fp = construct_peak(rt, param)
    print(fp)
    # print(np.asarray(param), np.asarray(fp))
    print(np.asarray(param))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rt, peaks)
    ax.plot(rt, fp, "r")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_peak_fit()
