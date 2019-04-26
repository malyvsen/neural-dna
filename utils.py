import numpy as np
import skimage
import matplotlib.pyplot as plt


def show(images, save_as=None, title=None):
    to_show = np.concatenate(images, 1)
    if save_as is not None:
        skimage.io.imsave(save_as, (to_show - np.min(to_show)) / (np.max(to_show) - np.min(to_show)))
    if title is not None:
        plt.title(title)
    plt.imshow(to_show, cmap='gray')
    plt.show()


def random_batch(data, labels, size):
    indices = np.random.choice(len(data), size)
    return data[indices], labels[indices]
