import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy.linalg as la
import scipy.ndimage as img


def read_img(path):
    imgF = iio.imread(path, mode="L").astype(np.float)
    return imgF


def binarize(imgF):
    imgD = np.abs(img.filters.gaussian_filter(imgF, sigma=0.5) - img.filters.gaussian_filter(imgF, sigma=1))
    return img.morphology.binary_closing(np.where(imgD < 0.1*imgD.max(), 0, 1))


def count_boxes_for_factor_s(imgD, s):
    h, w = imgD.shape
    box_size = int(s*h) # img is square

    counter = 0
    for i in range(0, h - box_size + 1, box_size):
        for j in range(0, w - box_size + 1, box_size):
            box = imgD[i:i+box_size, j:j+box_size]
            if np.max(box)==1:  # at least one pixel in the extracted box is a foreground pixel
                counter += 1
    return counter


def box_counting(img, title):
    # binarize image
    imgD = binarize(img)

    # set up the scaling factors
    h, w = imgD.shape # 512x512
    L = int(np.log2(h))
    l_indices = [i for i in range(1, L-1)]
    scaling_factors = [ 1/ (2**l) for l in l_indices]

    # count the boxes with foreground-pixels for each scaling factor
    n_values = [count_boxes_for_factor_s(imgD, s) for s in scaling_factors]
    print(n_values)
    x_values = [np.log2(1/s) for s in scaling_factors]
    y_values = [np.log2(n) for n in n_values]

    #### fit a line
    # set up point matrix X
    first_column = [x**0 for x in x_values]
    second_column = [x**1 for x in x_values]
    matX = np.vstack((first_column, second_column))

    # perform leastsquares
    (b, D) = optimal_w_with_lstsq(matX, y_values) # D = fractal dimension

    fitted_y_values = [D*x + b for x in x_values]
    plt.plot(x_values, fitted_y_values)
    plt.scatter(x_values, y_values)
    plt.xlabel("log(1/s)")
    plt.ylabel("log(n)")
    plt.title(f"{title}, slope D = {D}")
    plt.show()


def task1_3():
    img_path_1 = "lightning.png"
    img_path_2 = "tree.png"

    img_lightning = read_img(img_path_1)
    img_tree = read_img(img_path_2)

    box_counting(img_lightning, "Lightning")
    box_counting(img_tree, "Tree")


if __name__ == "__main__":
    task1_3()