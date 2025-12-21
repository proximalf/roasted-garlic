from typing import Dict

import numpy as np

def draw_square(img, value, size_ratio=0.5):
    h, w = img.shape[:2]
    size = int(min(h, w) * size_ratio)

    y0 = (h - size) // 2
    y1 = y0 + size
    x0 = (w - size) // 2
    x1 = x0 + size

    img[y0:y1, x0:x1] = value
    return img


img_gray8 = np.zeros((256, 256), dtype=np.uint8)
img_gray8 = draw_square(img_gray8, value=200)

img_gray16 = np.zeros((256, 256), dtype=np.uint16)
img_gray16 = draw_square(img_gray16, value=50000)


img_rgb8 = np.zeros((256, 256, 3), dtype=np.uint8)
img_rgb8 = draw_square(img_rgb8, value=(255, 0, 0))  # red square


img_rgb16 = np.zeros((256, 256, 3), dtype=np.uint16)
img_rgb16 = draw_square(img_rgb16, value=(65535, 0, 0))  # red square

img_rgba16 = np.zeros((256, 256, 4), dtype=np.uint16)
img_rgba16[..., 3] = 65535
img_rgba16 = draw_square(img_rgba16, value=(0, 65535, 0, 65535))

fimg_rgba16 = np.zeros((256, 256, 4), dtype=np.float16)
fimg_rgba16[..., 3] = 1.0
fimg_rgba16 = draw_square(fimg_rgba16, value=(0.3, 1.0, 0.3, 1.0))

fimg_rgb16 = np.zeros((256, 256, 3), dtype=np.float16)
fimg_rgb16 = draw_square(fimg_rgb16, value=(255, 155, 100))

fimg_rgba64 = np.zeros((256, 256, 4), dtype=np.float64)
fimg_rgba64[..., 3] = 1.0
fimg_rgba64 = draw_square(fimg_rgba64, value=(0.5, 1.0, 0.1, 1.0))

TEST_IMAGES: Dict[str, np.ndarray] = {
    "img_gray8": img_gray8, 
    "img_gray16": img_gray16, 
    "img_rgb8": img_rgb8, 
    "img_rgb16": img_rgb16, 
    "img_rgba16": img_rgba16, 
    "fimg_rgb16": fimg_rgb16,
    "fimg_rgba16": fimg_rgba16,
    "fimg_rgba64": fimg_rgba64,
    }