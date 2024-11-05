import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def niblack_threshold(image, window_size=15, k=-0.2):
    image = np.array(image)

    binary_image = np.zeros_like(image)

    half_window = window_size // 2

    for y in range(half_window, image.shape[0] - half_window):
        for x in range(half_window, image.shape[1] - half_window):
            window = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            m = np.mean(window)
            s = np.std(window)
            threshold = m + k * s
            if image[y, x] > threshold:
                binary_image[y, x] = 255
            else:
                binary_image[y, x] = 0

    return binary_image


def sauvola_threshold(image, window_size=15, k=0.5, R=128):
    image = np.array(image)

    binary_image = np.zeros_like(image)

    half_window = window_size // 2

    for y in range(half_window, image.shape[0] - half_window):
        for x in range(half_window, image.shape[1] - half_window):
            window = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            m = np.mean(window)
            s = np.std(window)
            threshold = m * (1 + k * ((s / R) - 1))
            if image[y, x] > threshold:
                binary_image[y, x] = 255
            else:
                binary_image[y, x] = 0

    return binary_image


def phansalkar_threshold(image, window_size=15, k=0.25, p=0.5, q=0.5, R=128):
    image = np.array(image)

    binary_image = np.zeros_like(image)

    half_window = window_size // 2

    for y in range(half_window, image.shape[0] - half_window):
        for x in range(half_window, image.shape[1] - half_window):

            window = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]

            m = np.mean(window)
            s = np.std(window)
            threshold = m * (1 + p * np.exp(-q * m) + k * ((s / R) - 1))
            if image[y, x] > threshold:
                binary_image[y, x] = 255
            else:
                binary_image[y, x] = 0

    return binary_image

def my_method(image, threshold=None):
    image = np.array(image)

    if threshold is None:
        threshold = np.mean(image)

    binary_image = np.zeros_like(image)

    binary_image[image > threshold] = 255
    binary_image[image <= threshold] = 0

    return binary_image

image_path = "image.jpg"
image = Image.open(image_path).convert('L')

niblack_image = niblack_threshold(image, k=0.5)
sauvola_image = sauvola_threshold(image, k=0.2, R=64)
phansalkar_image = phansalkar_threshold(image, window_size=15, k=0.2, p=0.5, q=0.5, R=64)
my_image = my_method(image)

Image.fromarray(niblack_image).save('niblack.jpg')
Image.fromarray(sauvola_image).save('sauvola.jpg')
Image.fromarray(phansalkar_image).save('phansalkar.jpg')
Image.fromarray(my_image).save('my_bin.jpg')

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(niblack_image, cmap='gray')
# plt.title('Niblack image')
# plt.imshow(sauvola_image, cmap='gray')
# plt.title('Sauvola image')
# plt.imshow(phansalkar_image, cmap='gray')
# plt.title('Phansalkar image')
# plt.axis('off')

# plt.show()