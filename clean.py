import os
import random
import shutil

# Paths
source_dir = "datasets/bfmc"  # Replace with your folder containing images and labels
train_img_dir = "images/train"
test_img_dir = "images/test"
train_lbl_dir = "images/labels/train"
test_lbl_dir = "images/labels/test"

# Create directories if they don't exist
for d in [train_img_dir, test_img_dir, train_lbl_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Collect valid images (those with a corresponding .txt file)
images = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]
valid_images = []
for img in images:
    txt_file = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(source_dir, txt_file)):
        valid_images.append(img)
    else:
        os.remove(os.path.join(source_dir, img))

# Shuffle and split
random.shuffle(valid_images)
split_idx = int(0.8 * len(valid_images))
train_images = valid_images[:split_idx]
test_images = valid_images[split_idx:]

# Move images and labels
for img_list, img_dest, lbl_dest in [
    (train_images, train_img_dir, train_lbl_dir),
    (test_images, test_img_dir, test_lbl_dir)
]:
    for img in img_list:
        txt_file = os.path.splitext(img)[0] + ".txt"
        shutil.move(os.path.join(source_dir, img), os.path.join(img_dest, img))
        shutil.move(os.path.join(source_dir, txt_file), os.path.join(lbl_dest, txt_file))
