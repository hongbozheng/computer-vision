#!/usr/bin/env python3


import argparse
import cv2
import matplotlib.pyplot
import numpy
import scipy.linalg
import time


def create_slideing_wins(mat, step, win_size) -> list:
    wins = []
    for k in range(win_size//2, mat.shape[0]-win_size//2, step):
        for i in range(win_size//2, mat.shape[1]-win_size//2, step):
            wins.append(((i, k), mat[k - win_size//2:k + win_size//2, i - win_size//2:i + win_size//2]))

    return wins


def ssd(x, y, win_size, offset, win, mat_1):
    min = 99999

    a = x - offset if x - offset > win_size//2 else win_size//2
    b = x + offset if x + offset < mat_1.shape[1] - win_size//2 else mat_1.shape[1] - win_size//2

    for i in range(a, b):
        tmp = numpy.sum(a=(win - mat_1[y - win_size//2:y + win_size//2, i - win_size//2:i + win_size//2]) ** 2)
        if tmp < min:
            min = tmp
            val = x - i

    return val


def sad(x, y, win_size, offset, win, mat_1):
    min = 99999

    a = x - offset if x - offset > win_size // 2 else win_size // 2
    b = x + offset if x + offset < mat_1.shape[1] - win_size // 2 else mat_1.shape[1] - win_size // 2

    for i in range(a, b):
        tmp = numpy.abs(win - mat_1[y - win_size // 2:y + win_size // 2, i - win_size // 2:i + win_size // 2])
        tmp = numpy.sum(a=tmp)
        if tmp < min:
            min = tmp
            val = x - i

    return val


def ncc(x, y, win_size, offset, win, mat_1):
    max = -99999

    a = x - offset if x - offset > win_size // 2 else win_size // 2
    b = x + offset if x + offset < mat_1.shape[1] - win_size // 2 else mat_1.shape[1] - win_size // 2

    win = win - numpy.mean(a=win, axis=0)

    for i in range(a, b):
        mu = numpy.mean(a=mat_1[y - win_size // 2:y + win_size // 2, i - win_size // 2:i + win_size // 2], axis=0)
        mat = mat_1[y - win_size // 2:y + win_size // 2, i - win_size // 2:i + win_size // 2] - mu
        tmp = numpy.sum(a=(win / scipy.linalg.norm(a=win)) * (mat / scipy.linalg.norm(a=mat)))
        if tmp > max:
            max = tmp
            val = x - i

    return val


def main():
    parser = argparse.ArgumentParser(
        prog="binocular_stereo",
        description="window-based stereo matching algorithm for rectified stereo images pair")
    valid_algos = ["SSD", "SAD", "NCC"]
    parser.add_argument("--algorithm", "-a", type=str, required=True, choices=valid_algos,
                        help="matching algorithm")
    parser.add_argument("--step", "-s", type=int, required=True, help="step of window")
    parser.add_argument("--win_size", "-w", type=int, required=True, help="window size")
    parser.add_argument("--offset", "-o", type=int, required=True, help="offset")

    args = parser.parse_args()
    algo = args.algorithm
    step = args.step
    win_size = args.win_size
    offset = args.offset

    filepaths = ["data/moebius1.png", "data/moebius2.png"]
    # filepaths = ["data/tsukuba1.jpg", "data/tsukuba2.jpg"]

    mat_0 = cv2.imread(filename=filepaths[0], flags=cv2.IMREAD_GRAYSCALE).astype(dtype=numpy.float64)
    mat_1 = cv2.imread(filename=filepaths[1], flags=cv2.IMREAD_GRAYSCALE).astype(dtype=numpy.float64)

    # downsize
    mat_0 = cv2.resize(src=mat_0, dsize=(mat_0.shape[1]//4, mat_0.shape[0]//4)).astype(dtype=numpy.float64)
    mat_1 = cv2.resize(src=mat_1, dsize=(mat_1.shape[1]//4, mat_1.shape[0]//4)).astype(dtype=numpy.float64)

    # upsample
    # mat_0 = cv2.resize(src=mat_0, dsize=(int(mat_0.shape[0]*1.2), int(mat_0.shape[1]*1.2))).astype(dtype=numpy.float64)
    # mat_1 = cv2.resize(src=mat_1, dsize=(int(mat_1.shape[0]*1.2), int(mat_1.shape[1]*1.2))).astype(dtype=numpy.float64)

    mat_0 /= 255.0
    mat_1 /= 255.0

    wins = create_slideing_wins(mat=mat_0, step=step, win_size=win_size)

    disparity_map = numpy.zeros(shape=mat_1.shape, dtype=numpy.uint8)

    start_time = time.time()

    if algo == "SSD":
        for (x, y), win in wins:
            val = abs(ssd(x=x, y=y, win_size=win_size, offset=offset, win=win, mat_1=mat_1)) / offset
            disparity_map[y, x] = val * 255
    elif algo == "SAD":
        for (x, y), win in wins:
            val = abs(sad(x=x, y=y, win_size=win_size, offset=offset, win=win, mat_1=mat_1)) / offset
            disparity_map[y, x] = val * 255
    elif algo == "NCC":
        for (x, y), win in wins:
            val = abs(ncc(x=x, y=y, win_size=win_size, offset=offset, win=win, mat_1=mat_1)) / offset
            disparity_map[y, x] = val * 255
    else:
        print("[ERROR]: Invalid algorithm.")

    end_time = time.time()

    print(f"[INFO]: Total run-time {end_time-start_time}")

    matplotlib.pyplot.imshow(X=disparity_map, cmap="gray")
    matplotlib.pyplot.show()

    return


if __name__ == "__main__":
    main()