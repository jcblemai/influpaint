
import numpy as np

# These transform applies to a numpy object with dimensions
#  (feature, date, place)


def transform_randomscale(image, max, min):
    import random

    scale = random.uniform(min, max)  # TODO should not be uniform !!!
    return image * scale


def transform_channelwisescale(
    image, scale
):  # TODO write for three channel like it was above
    return image * scale


def transform_channelwisescale_inv(image, scale):
    return image / scale


def transform_sqrt(image):
    return np.sqrt(image)


def transform_sqrt_inv(image):
    return image ** 2


def transform_shift(image, shift=-1):
    return image + shift


def transform_shift_inv(image, shift=-1):
    return image - shift


def transform_rollintime(image, shift):
    r_val = np.roll(image, shift=shift, axis=1)
    return r_val


def transform_random_rollintime(image, min_shift, max_shift):
    import random

    shift = random.randint(min_shift, max_shift)
    return transform_rollintime(image, shift)


def transform_random_padintime(image, min_shift, max_shift, neutral_value=0):
    import random

    shift = random.randint(min_shift, max_shift)
    r_val = transform_rollintime(image, shift)
    if shift >= 0:
        r_val[:, :shift] = neutral_value
    else:
        r_val[:, shift:] = neutral_value

    return r_val


def transform_randomnoise(image, sigma=0.2):
    mu = 1
    return image * np.random.normal(mu, sigma, image.shape)

def transform_skewednoise(image, scale=0.4, a=-1):
    from scipy.stats import skewnorm
    r = skewnorm.rvs(loc=1, scale=scale, a=a, size=image.shape)
    r[r < 0] = 0
    return image * r

def transform_poisson(image):
    return np.random.poisson(image)
