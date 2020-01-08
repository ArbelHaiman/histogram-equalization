import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
'''
Author: Arbel Haiman
'''
# The implementation is according to the book Digital image proccesing 3rd edition by Rafael C. Gonzalez, chapter 3.

# A function that executes histogram equalization, on a given image.
# This method improves the contrast in images. The regular, meaning the global
# HE is very fast, comparing to the LHE, which is very slow, but improves the contrast even more.
def histogram_equalization(image):
    """
    applying GHE on the image.
    :param image: The image to apply HE on.
    :return: The imporoved image
    """
    rows = image.shape[0]
    cols = image.shape[1]
    MN = rows * cols
    n_k_for_r = np.zeros((256,1))
    n_k_for_s = np.zeros((256, 1))

    # calculating the number of pixels corresponding to each intensity. the values are stored in n_k.
    # This is the histogram.
    # n_k[k] stores the number of pixels with intensity k, in the input image.
    for i in range(0,rows):
        for j in range(0,cols):
            n_k_for_r[image[i,j]] += 1

    # calculating the rational histogram
    pdf_r = pdf(n_k_for_r, MN)

    # calculating the new intensity for each intensity, meaning the mapping function.
    s_k = s_func(pdf_r)

    # transform the image and get the enhanced image.
    improved_image = mapping_function(s_k, image)
    return improved_image

# A function that perform local histogram equalization
def local_histogram_equalization(image, root_size_of_neighbourhood=3):
    """
    performes LHE on the given image with the given neighbourhood size.
    :param image: The image to apply LHE on.
    :param root_size_of_neighbourhood: The root size of the neighbourhood for the algorithm.
    :return: The improved image.
    """
    print('please wait patiently...')
    # padding the image, so we can calculate on the image's borders as well.
    image = cv2.copyMakeBorder(image, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REFLECT)

    rows = image.shape[0]
    cols = image.shape[1]
    output = np.zeros((rows, cols))
    MN = root_size_of_neighbourhood ** 2
    n_k_for_r = np.zeros((256, 1))

    # going through the image and calculate a histogram for each neighbourhood.
    # Then, calculate the new value of the center pixel, according to the neighbourhood's histogram.

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighbourhood = np.array(image[i-1: i+2, j-1: j+2])

            # neighbourhood histogram
            for m in range(0, root_size_of_neighbourhood):
                for n in range(0, root_size_of_neighbourhood):
                    n_k_for_r[neighbourhood[m, n]] += 1

            # neighbourhood rational histogram.
            local_pdf_r = pdf(n_k_for_r, MN)

            # mapping function.
            local_s_k = s_func(local_pdf_r)

            # calculate the new value.
            output[i, j] = local_s_k[image[i, j]]

            n_k_for_r = np.zeros((256, 1))

    # making sure the image is in the right type for showing it with opencv.
    output = output.astype(np.uint8)

    # cutting off padding
    output = output[1: output.shape[0] - 1, 1: output.shape[1] - 1]

    return output

# a function which calculates the pdf of an image, meaning the rational histogram.
# it returns an array of length 256, for which each cell
# represents an intensity corresponding to its index.
# the value stored in each cell is the fraction of pixels
# which has this intensity in the input image, out of all pixels in the image.
def pdf(intensity_array_n_k, MN):
    """
    Calculates the pdf of the image.
    :param intensity_array_n_k: The histogram of the image.
    :param MN: Total number of pixels in the image.
    :return:
    """
    pdf = np.zeros((256,1))
    return intensity_array_n_k/MN

# a function which computes the s function of a given pdf_r.
# the function returns an array s_k, which stores the values of s.
# s_k[k] stores the value of s_k, according to the formula 3.3-8 in the book,
# that is, the sum of all cells in pdf_r from cell 0 to cell k, including.
def s_func(pdf_r):
    """

    :param pdf_r: The rational histogram to calculate s on.
    :return: A mapping function for the image.
    """
    s_k = np.zeros((256,1))
    s_k[0] = pdf_r[0]
    for k in range(1, s_k.shape[0]):
        s_k[k] = pdf_r[k]+s_k[k-1]

    s_k = 255 * s_k
    return s_k


def mapping_function(map_func_array, image):
    """
    Calculates the transformed image.
    :param map_func_array: The s array, meaning the mapping function, which specifies for each intensity,
        the transformed intensity.
    :param image: The image to enhance.
    :return: The final improved image.
    """

    rows = image.shape[0]
    cols = image.shape[1]
    improved_image = np.zeros((rows , cols))
    for i in range(0, rows):
        for j in range(0, cols):
            improved_image[i,j] = map_func_array[image[i,j]]

    # cutting off the padding that was added
    image = image[1: image.shape[0], 1: image.shape[1]]
    # making sure the image is in the right type for showing it with opencv.
    improved_image = improved_image.astype(np.uint8)
    return improved_image


# The main program
neighbourhood_size = 3

# original image
image_path = 'embedded_squares.JPG'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('original', image)
cv2.waitKey()

# global histogram equalization
time1 = time.time()
ghe_out = histogram_equalization(image=image)
time2 = time.time()
print('GHE function took ' + str(time2-time1) + ' seconds')
cv2.imshow('global HE', ghe_out)
cv2.waitKey()

time1 = time.time()
lhe_out = local_histogram_equalization(image=image, root_size_of_neighbourhood=neighbourhood_size)
time2 = time.time()
print('LHE function took ' + str(time2-time1) + ' seconds')
cv2.imshow('LHE with neighbourhood of size ' + str(neighbourhood_size) + ': ', lhe_out)
cv2.waitKey()