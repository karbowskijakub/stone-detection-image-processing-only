"""
Module for gamma correction of CT images of kidneys.

"""

import os
import numpy as np
from skimage import exposure, filters, io
from skimage.util import img_as_ubyte, img_as_float
from typing import Union, Tuple, List
import matplotlib.pyplot as plt


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma has to be a positive number.")
    return exposure.adjust_gamma(image, gamma)


def adaptive_gamma_correction(
    image: np.ndarray, target_mean: float = 0.5
) -> np.ndarray:
    image_float = img_as_float(image)
    if target_mean > 1.0:
        target_mean = target_mean / 255.0
    current_mean = np.mean(image_float)
    gamma = np.log(target_mean) / np.log(current_mean) if current_mean > 0 else 1.0
    gamma = np.clip(gamma, 0.3, 3.0)
    result = exposure.adjust_gamma(image_float, gamma)
    return img_as_ubyte(result) if image.dtype == np.uint8 else result


def clahe_gamma_correction(
    image: np.ndarray,
    clip_limit: float = 0.03,
    kernel_size: Union[int, Tuple[int, int]] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    image_float = img_as_float(image)
    enhanced = exposure.equalize_adapthist(
        image_float, kernel_size=kernel_size, clip_limit=clip_limit
    )
    if gamma != 1.0:
        enhanced = exposure.adjust_gamma(enhanced, gamma)
    return img_as_ubyte(enhanced) if image.dtype == np.uint8 else enhanced


def sigmoid_correction(
    image: np.ndarray, cutoff: float = 0.5, gain: float = 10
) -> np.ndarray:
    image_float = img_as_float(image)
    corrected = exposure.adjust_sigmoid(image_float, cutoff=cutoff, gain=gain)
    return img_as_ubyte(corrected) if image.dtype == np.uint8 else corrected


def log_correction(image: np.ndarray, gain: float = 1.0) -> np.ndarray:
    image_float = img_as_float(image)
    corrected = exposure.adjust_log(image_float, gain=gain)
    return img_as_ubyte(corrected) if image.dtype == np.uint8 else corrected


def contrast_stretching(
    image: np.ndarray,
    in_range: str = "image",
    percentiles: Tuple[float, float] = (2, 98),
) -> np.ndarray:
    image_float = img_as_float(image)
    if in_range == "image" and percentiles:
        p_low, p_high = np.percentile(image_float, percentiles)
        in_range = (p_low, p_high)
    stretched = exposure.rescale_intensity(image_float, in_range=in_range)
    return img_as_ubyte(stretched) if image.dtype == np.uint8 else stretched


def auto_gamma_for_ct(image: np.ndarray, enhance_bright: bool = True) -> np.ndarray:
    image_float = img_as_float(image)
    hist, bins = exposure.histogram(image_float)
    cumsum = np.cumsum(hist) / np.sum(hist)
    median_idx = np.searchsorted(cumsum, 0.5)
    median_value = bins[median_idx]
    if enhance_bright:
        gamma = 0.7 if median_value < 0.4 else 0.9 if median_value < 0.6 else 1.1
    else:
        gamma = 1.0 if median_value > 0.4 else 0.8
    result = exposure.adjust_gamma(image_float, gamma)
    return img_as_ubyte(result) if image.dtype == np.uint8 else result


def multi_scale_retinex(
    image: np.ndarray, sigmas: List[float] = [15, 80, 250]
) -> np.ndarray:
    image_float = np.maximum(img_as_float(image), 1e-6)
    retinex = np.zeros_like(image_float)
    for sigma in sigmas:
        blurred = np.maximum(
            filters.gaussian(image_float, sigma=sigma, mode="reflect"), 1e-6
        )
        retinex += np.log(image_float) - np.log(blurred)
    retinex /= len(sigmas)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
    return img_as_ubyte(retinex) if image.dtype == np.uint8 else retinex


def histogram_equalization(
    image: np.ndarray, adaptive: bool = True, **kwargs
) -> np.ndarray:
    image_float = img_as_float(image)
    if adaptive:
        result = exposure.equalize_adapthist(image_float, **kwargs)
    else:
        result = exposure.equalize_hist(image_float)
    return img_as_ubyte(result) if image.dtype == np.uint8 else result


def process_image(file_path: str, processed_dir: str):
    """
    Processes the image using all methods.
    """
    os.makedirs(processed_dir, exist_ok=True)
    img = io.imread(file_path, as_gray=True)
    if img.dtype != "uint8":
        img = img_as_ubyte(img)

    results = {
        "gamma_05": gamma_correction(img, 0.5),
        "gamma_15": gamma_correction(img, 1.5),
        "adaptive_gamma": adaptive_gamma_correction(img),
        "clahe_enhanced": clahe_gamma_correction(img, gamma=0.8),
        "auto_ct": auto_gamma_for_ct(img),
        "sigmoid": sigmoid_correction(img, cutoff=0.5, gain=10),
        "stretched": contrast_stretching(img),
        "msr": multi_scale_retinex(img),
    }

    # Save results
    # for name, result_img in results.items():
    #     save_path = os.path.join(processed_dir, f'{name}.png')
    #     io.imsave(save_path, result_img)
    #     print(f"Zapisano {save_path}")

    # Visualization
    all_images = [("Original image", img)] + [(k, v) for k, v in results.items()]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for ax, (title, image_data) in zip(axes.flat, all_images):
        ax.imshow(image_data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, 'gamma_comparison.png'), dpi=150)
    print("Saved comparison to gamma_comparison.png")


if __name__ == "__main__":
    # Set catalog manualy
    project_root = r"c:\Project\stone-detection-image-processing-only"

    raw_file = os.path.join(
        project_root, "data", "raw", "1-3-46-670589-33-1-63709079072689291800001-4862263206834957747_png_jpg.rf.cd218d4ed600ad3b24e4af71077096bd.jpg"
    )

    processed_dir = os.path.join(project_root, "data", "processed")

    process_image(raw_file, processed_dir=processed_dir)
