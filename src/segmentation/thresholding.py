"""
Module for thresholding CT images of kidneys.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.util import img_as_ubyte


def global_threshold(image: np.ndarray) -> np.ndarray:
    thresh = np.mean(image)
    return image > thresh


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(image)
    return image > thresh


def adaptive_threshold(
    image: np.ndarray, block_size: int = 35, offset: float = 0.0
) -> np.ndarray:
    thresh = filters.threshold_local(image, block_size, offset=offset)
    return image > thresh


def yen_threshold(image: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_yen(image)
    return image > thresh


def li_threshold(image: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_li(image)
    return image > thresh


def process_image(file_path: str, processed_dir: str):
    os.makedirs(processed_dir, exist_ok=True)

    img = io.imread(file_path, as_gray=True)
    img = img_as_ubyte(img)

    results = {
        "global": global_threshold(img),
        "otsu": otsu_threshold(img),
        "adaptive": adaptive_threshold(img),
        "yen": yen_threshold(img),
        "li": li_threshold(img),
    }

    all_images = [("original", img)] + list(results.items())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, image_data) in zip(axes.flat, all_images):
        ax.imshow(image_data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(processed_dir, "thresholding_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    project_root = r"c:\Project\stone-detection-image-processing-only"

    raw_file = os.path.join(
        project_root, "data", "raw", "1-3-46-670589-33-1-63709079072689291800001-4862263206834957747_png_jpg.rf.cd218d4ed600ad3b24e4af71077096bd.jpg"
    )

    processed_dir = os.path.join(project_root, "data", "processed")

    process_image(raw_file, processed_dir)
