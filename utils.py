import numpy as np
import skimage
import matplotlib.pyplot as plt


def show(images, save_as=None, title=None):
    to_show = np.clip(np.concatenate(images, 1), 0.0, 1.0)
    if save_as is not None:
        skimage.io.imsave(save_as, to_show)
    if title is not None:
        plt.title(title)
    plt.imshow(to_show)
    plt.show()


def random_batch(data, size):
    return data[np.random.choice(len(data), size)]
