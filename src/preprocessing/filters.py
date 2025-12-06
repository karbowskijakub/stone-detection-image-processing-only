"""
Module for comparing noise filtering on CT images of kidneys: median vs Gaussian.
"""

import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_ubyte, img_as_float
from skimage.filters import gaussian, median
from skimage.morphology import disk

def apply_gaussian_filter(image, sigma=1.0):
    image_float = img_as_float(image)
    filtered = gaussian(image_float, sigma=sigma)
    return img_as_ubyte(filtered)

def apply_median_filter(image, disk_size=3):
    image_float = img_as_float(image)
    filtered = median(image_float, disk(disk_size))
    return img_as_ubyte(filtered)

def compare_filters(file_path, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

    img = io.imread(file_path, as_gray=True)
    img_gauss = apply_gaussian_filter(img, sigma=1.0)
    img_median = apply_median_filter(img, disk_size=3)

    all_images = [
        ("original", img),
        ("Gaussian filter", img_gauss),
        ("Median filter", img_median)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, image_data) in zip(axes.flat, all_images):
        ax.imshow(image_data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, "filter_comparison.png"), dpi=150, bbox_inches="tight")
    # io.imsave(os.path.join(processed_dir, "gaussian_filtered.png"), img_gauss)
    # io.imsave(os.path.join(processed_dir, "median_filtered.png"), img_median)

if __name__ == "__main__":
    project_root = r"c:\Project\stone-detection-image-processing-only"
    raw_file = os.path.join(
        project_root, "data", "raw",
        "1-3-46-670589-33-1-63709079072689291800001-4862263206834957747_png_jpg.rf.cd218d4ed600ad3b24e4af71077096bd.jpg"
    )
    processed_dir = os.path.join(project_root, "data", "processed")
    compare_filters(raw_file, processed_dir)
