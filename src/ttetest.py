import numpy as np


def srgb_decode(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def srgb_encode(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

def emulate_browser(variant_u8, M, u, scale=1.0):
    """What the SVG filter should compute: variant (8-bit sRGB) -> adjusted 8-bit sRGB."""
    h, w, _ = variant_u8.shape
    linear = srgb_decode(variant_u8.reshape(-1, 3) / 255.0)

    A = M @ np.diag(u) @ np.linalg.inv(M)
    adjusted = linear @ A.T            # same as applying A to every pixel
    adjusted = np.clip(adjusted * scale, 0, 1)

    return np.round(srgb_encode(adjusted) * 255).reshape(h, w, 3).astype(np.uint8)

