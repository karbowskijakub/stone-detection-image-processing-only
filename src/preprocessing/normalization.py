"""
Module for normalization and contrast enhancement of CT images of kidneys.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_ubyte, img_as_float

def normalize_image(image: np.ndarray) -> np.ndarray:
    image_float = img_as_float(image)
    return (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float))

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    return exposure.equalize_adapthist(image)

def process_image(file_path: str, processed_dir: str):
    os.makedirs(processed_dir, exist_ok=True)

    img = io.imread(file_path, as_gray=True)
    img_norm = normalize_image(img)
    img_contrast = enhance_contrast(img_norm)
    img_final = img_as_ubyte(img_contrast)

    all_images = [("original", img), ("normalized", img_norm), ("contrast", img_final)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, image_data) in zip(axes.flat, all_images):
        ax.imshow(image_data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(processed_dir, "preprocessing_comparison.png"), dpi=150, bbox_inches="tight")
    # io.imsave(os.path.join(processed_dir, "preprocessed_image.png"), img_final)

if __name__ == "__main__":
    project_root = r"c:\Project\stone-detection-image-processing-only"
    raw_file = os.path.join(
        project_root, "data", "raw",
        "1-3-46-670589-33-1-63709079072689291800001-4862263206834957747_png_jpg.rf.cd218d4ed600ad3b24e4af71077096bd.jpg"
    )
    processed_dir = os.path.join(project_root, "data", "processed")
    process_image(raw_file, processed_dir)
