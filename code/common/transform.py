import numpy as np
from PIL import Image


def tranform_response(response, new_width=84, new_height=84):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255. / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d,
        (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((new_width, new_height)).convert('L'))

    return im_final


def transform_response_array(response_array, transform_configs=None):
    if transform_configs = None:
        return [transform_response(item) for item in response_array]
    else:
        return [transform_response(item, width, height)
                for item, (width, height) in
                zip(reponse_array, transform_configs)]


def np_transform_reponse_array(response_array):
    return np.array(transform_reponse_array(response_array), dtype=np.float32)


def transform_input(responses):
    return transform_response(responses[0])


