import io

import numpy as np
from PIL import Image


def _maybe_decode_encoded(images: np.ndarray) -> np.ndarray:
    """Decode images that arrived as encoded bytes (e.g. JPEG/PNG) instead of raw arrays.

    Some clients send compressed image bytes over the wire to save bandwidth. msgpack
    carries those as a numpy array with a bytes/str/object dtype (e.g. ``|S74063``) rather
    than a ``uint8`` H×W×C array. Decode them back here so the rest of the resize path can
    treat every image uniformly. Arrays that are already numeric are returned unchanged.
    """
    if images.dtype.kind not in ("S", "U", "O"):
        return images

    flat = images.reshape(-1)
    decoded = []
    for item in flat:
        if isinstance(item, np.ndarray):
            item = item.item()
        if isinstance(item, str):
            item = item.encode("latin-1")
        decoded.append(np.asarray(Image.open(io.BytesIO(item)).convert("RGB")))
    decoded = np.stack(decoded)
    # Drop the scalar element axis and restore any leading (batch) dims.
    return decoded.reshape(*images.shape, *decoded.shape[1:])


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # Decode encoded byte blobs (e.g. JPEG/PNG) into raw uint8 arrays first.
    images = _maybe_decode_encoded(images)

    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
