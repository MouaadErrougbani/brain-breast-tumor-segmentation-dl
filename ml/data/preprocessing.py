import os 
import shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter


PATH_BRAIN_DATA = "./data/raw/brain-tumor-dataset-includes-the-mask-and-images/data/data"
PATH_BREAST_DATA= "./data/raw/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
OUTPUT_DIR_PREPROCESSING = "./data/processed"
SEED = 42
# We will randomly select 647 Brain Tumor images to match the number of Breast Tumor images
# (210 malignant and 437 benign) and create a balanced dataset.
NB_IMAGES_FOR_EACH_TUMOR_TYPE = 647
SIZE = (512, 512)

def separate_and_move_breast_images_masks(source_dir, destination_dir): 
    images = os.listdir(source_dir)
    images_dir = os.path.join(destination_dir, "images")
    masks_dir = os.path.join(destination_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    for img in images: 
        dest_img = img.replace('(', '').replace(')', '').replace(' ', '_').replace("benign", "breast_0") \
        .replace("malignant", "breast_1")
        if "_mask" in img: 
            shutil.copy(os.path.join(source_dir, img), os.path.join(masks_dir, dest_img))
        else: 
            shutil.copy(os.path.join(source_dir, img), os.path.join(images_dir, dest_img))



def randomly_select_and_move_brain_images_masks(source_dir, destination_dir): 
    source_images = os.path.join(source_dir, "images")
    source_masks = os.path.join(source_dir, "masks")

    destination_images = os.path.join(destination_dir, "images")
    destination_masks = os.path.join(destination_dir, "masks")

    os.makedirs(destination_images, exist_ok=True)
    os.makedirs(destination_masks, exist_ok=True)
    
    images = os.listdir(source_images)
    np.random.seed(SEED)

    images_selected = np.random.choice(
        images,
        size=NB_IMAGES_FOR_EACH_TUMOR_TYPE,
        replace=False
    )
    for img in images_selected: 
        name, ext = os.path.splitext(img)
        shutil.copy(os.path.join(source_images, img), os.path.join(destination_images, f'brain_{img}'))
        shutil.copy(os.path.join(source_masks, img), os.path.join(destination_masks, f'brain_{name}_mask{ext}'))

def get_images_masks(path):
    images = os.listdir(path+"/images")
    masks = os.listdir(path+"/masks")
    return images, masks

def info(images, masks): 
    nb_all_images = len(images)
    nb_all_masks = len(masks)
    nb_brain_images = len([img  for img in images if "brain" in img])
    nb_brain_masks = len([img for img in masks if "brain" in img])
    nb_breast_images = len([img for img in images if "breast" in img])
    nb_breast_masks = len([img for img in masks if "breast" in img ])

    return nb_all_images, nb_all_masks, nb_brain_images, nb_brain_masks, nb_breast_images, nb_breast_masks


def valide_masks(images, masks) : 
    for image in images: 
        if f'{image.split(".")[0]}_mask.png' not in masks: 
            print("This image '", image, "' not the mask.")
            break
    else :
        print("All images are the mask.")

def resize_image(path):
    with Image.open(path) as im:
        im = im.convert("L")
        im = im.resize(SIZE, Image.Resampling.BICUBIC)
        im.save(path)

def resize_mask(path):
    with Image.open(path) as m:
        m = m.convert("L").resize(SIZE, Image.Resampling.NEAREST)
        arr = (np.array(m) > 127).astype(np.uint8) * 255
        Image.fromarray(arr).save(path)

def fusion_masks(masks_info):
    for mask_name, count in masks_info.items():
        mask_path = os.path.join(
            OUTPUT_DIR_PREPROCESSING,
            "masks",
            mask_name
        )

        # Load the first mask
        with Image.open(mask_path).convert("L") as mask:
            merged_mask = np.array(mask, dtype=np.uint8)

        name, ext = os.path.splitext(mask_name)

        # Merge the remaining masks
        for i in range(1, count):
            mask_path = os.path.join(
                OUTPUT_DIR_PREPROCESSING,
                "masks",
                f"{name}_{i}{ext}"
            )

            with Image.open(mask_path).convert("L") as mask:
                current_mask = np.array(mask, dtype=np.uint8)

            merged_mask = np.maximum(merged_mask, current_mask)

        # Convert to binary mask
        merged_mask = (merged_mask > 0).astype(np.uint8) * 255

        # Save the merged mask
        Image.fromarray(merged_mask).save(
            os.path.join(OUTPUT_DIR_PREPROCESSING, "masks", mask_name)
        )

        # Remove additional masks
        for i in range(1, count):
            mask_path = os.path.join(
                OUTPUT_DIR_PREPROCESSING,
                "masks",
                f"{name}_{i}{ext}"
            )

            if os.path.exists(mask_path):
                os.remove(mask_path)

def nb_masks_for_image(masks):
    nb_masks = {}
    path_masks= {}
    for mask in masks : 
        img = mask.split("_mask")[0]
        nb_masks[img] =  nb_masks[img] + 1 if img in nb_masks.keys() else  1 
    for key, val in nb_masks.items() : 
        if val > 1 :
            path_masks[f'{key}_mask.png'] = val 
    return path_masks     


def preprocessing():

    # Breast benign
    separate_and_move_breast_images_masks(os.path.join(PATH_BREAST_DATA, "benign"), OUTPUT_DIR_PREPROCESSING)

    # Breast malignant
    separate_and_move_breast_images_masks(os.path.join(PATH_BREAST_DATA, "malignant"), OUTPUT_DIR_PREPROCESSING)

    # Brain
    randomly_select_and_move_brain_images_masks(os.path.join(PATH_BRAIN_DATA), OUTPUT_DIR_PREPROCESSING)

    images, masks = get_images_masks(OUTPUT_DIR_PREPROCESSING)
    nb_all_images, nb_all_masks, nb_brain_images, nb_brain_masks, nb_breast_images, nb_breast_masks = info(images, masks)
    valide_masks(images, masks)

    resize_image(os.path.join(OUTPUT_DIR_PREPROCESSING, "images"))
    resize_mask(os.path.join(OUTPUT_DIR_PREPROCESSING, "masks"))

    path_masks = nb_masks_for_image(masks)
    fusion_masks(path_masks)


if __name__ == "__main__":
    preprocessing()