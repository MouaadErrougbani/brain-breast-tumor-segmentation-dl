import os
import shutil
from sklearn.model_selection import train_test_split 

OUTPUT_DIR_DATASET = "./data/splits"
INPUT_DIR_DATASET = "./data/processed"
SEED = 42

def get_images_masks(path):
    images = os.listdir(path+"/images")
    masks = os.listdir(path+"/masks")
    return images, masks

def get_labels(X): 
    return ["brain" if "brain" in x else "breast" for x in X]

def copy_split(images, dest_dir): 
    source_images = os.path.join(INPUT_DIR_DATASET, "images")
    source_masks = os.path.join(INPUT_DIR_DATASET, "masks")
    dest_images = os.path.join(OUTPUT_DIR_DATASET, dest_dir, "images") 
    dest_masks = os.path.join(OUTPUT_DIR_DATASET, dest_dir, "masks") 

    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_masks, exist_ok=True)
    for img in images : 
        name, ext = os.path.splitext(img)
        shutil.copy(os.path.join(source_images, img), os.path.join(dest_images, img))
        mask = f'{name}_mask{ext}'
        shutil.copy(os.path.join(source_masks, mask), os.path.join(dest_masks, mask))


def split():

    images, masks = get_images_masks(INPUT_DIR_DATASET)
    labels = get_labels(images)

    train, temp = train_test_split(
        images,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    val, test = train_test_split(
        temp, 
        test_size=0.5,
        random_state=SEED, 
        stratify=get_labels(temp)
    )

    labels.count("brain"), labels.count("breast"), len(train), len(val), len(test)

    copy_split(train, "train")
    copy_split(val, "val")
    copy_split(test, "test")

if __name__ == "__main__":
    split()