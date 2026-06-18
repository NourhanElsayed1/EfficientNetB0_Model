
!pip install gdown
import gdown

url = "https://drive.google.com/file/d/1Za-V2nJDPa9qiTRU8RpAd9GAaW-LPg1z/view?usp=drive_link"
output = "data.zip"

gdown.download(url, output, quiet=False, fuzzy=True)


import zipfile
import os

zip_path = "/content/data.zip"
extract_path = "/content/Data"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted to:", extract_path)


import os
import shutil
from sklearn.model_selection import train_test_split

data_root = "/content/Data/Data/NEU-DET"

train_dir = os.path.join(data_root, "train", "images")
val_dir   = os.path.join(data_root, "validation", "images")
test_dir  = os.path.join(data_root, "test")


# 2️⃣ أنشئ فولدر test جديد فارغ
os.makedirs(test_dir, exist_ok=True)

# 3️⃣ دالة لنسخ الملفات إلى test
def copy_to_test_from_list(files, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for f, src_path in files:
        shutil.copy2(src_path, os.path.join(target_dir, f))

# 4️⃣ معالجة كل class
for cls_name in os.listdir(train_dir):
    cls_train_path = os.path.join(train_dir, cls_name)
    cls_val_path   = os.path.join(val_dir, cls_name)

    # جمع كل الصور من train + validation
    all_files = []
    for folder_path in [cls_train_path, cls_val_path]:
        if os.path.exists(folder_path):
            all_files += [(f, os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith((".jpg",".png"))]

    if not all_files:
        print(f"No images found for class {cls_name}. Skipping.")
        continue

    # اختر 15% من الصور لكل class
    test_files, _ = train_test_split(all_files, test_size=0.15, random_state=42)

    # انسخها لـ test
    target_cls_dir = os.path.join(test_dir, cls_name)
    copy_to_test_from_list(test_files, target_cls_dir)

    print(f"Copied {len(test_files)} images to test for class {cls_name}")

print("\n✅ Done! Test folder has been updated with ~15% of each class.")


import os

def print_dataset_summary(data_root):
    groups = {
        "train": os.path.join(data_root, "train", "images"),
        "validation": os.path.join(data_root, "validation", "images"),
        "test": os.path.join(data_root, "test")
    }

    all_classes = set()
    for group, path in groups.items():
        if os.path.exists(path):
            for cls in os.listdir(path):
                cls_path = os.path.join(path, cls)
                if os.path.isdir(cls_path):
                    all_classes.add(cls)

    print("\nDataset Summary:")
    print(f"{'Class':<20} {'Train':<10} {'Validation':<12} {'Test':<8} {'Test %':<8}")
    print("-"*60)

    for cls in sorted(all_classes):
        counts = {}
        for group, path in groups.items():
            cls_path = os.path.join(path, cls)
            if os.path.exists(cls_path):
                counts[group] = len([f for f in os.listdir(cls_path) if f.endswith((".jpg",".png"))])
            else:
                counts[group] = 0
        total = counts["train"] + counts["validation"] + counts["test"]
        test_percent = counts["test"] / total * 100 if total > 0 else 0
        print(f"{cls:<20} {counts['train']:<10} {counts['validation']:<12} {counts['test']:<8} {test_percent:>6.2f}%")

# استخدم المسار الصحيح
print_dataset_summary("/content/Data/Data/NEU-DET")




#اللي نجح في التدريب والدقة 94% 

# -----------------------------
# FULL PIPELINE (FIXED VERSION FOR COLAB)
# -----------------------------
print("FULL Pipeline: Collect, Clean, Split, Prepare DataLoaders, Train EfficientNetB0")

import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# -----------------------------
# PARAMETERS
# -----------------------------
DATA_DIR = Path("/content/Data/Data/NEU-DET")
CLEANED_DIR = Path("/content/Data/cleaned_all_images")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# -----------------------------
# PHASE 1: Collect ALL images
# -----------------------------
print("PHASE 1: Collecting ALL images...")

all_image_files = []
for folder in [DATA_DIR / 'train' / 'images', DATA_DIR / 'validation' / 'images']:
    if folder.exists():
        for cls_folder in folder.iterdir():
            if cls_folder.is_dir():
                all_image_files.extend(cls_folder.glob("*.jpg"))

print(f"Total images found: {len(all_image_files)}")

# -----------------------------
# Cleaning
# -----------------------------
print("Cleaning and standardizing images...")

def clean_and_standardize_image(image_path, target_size=IMG_SIZE):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size)

        # CLAHE
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        normalized = enhanced.astype(np.float32) / 255.0
        return normalized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def preprocess_all_images(image_files, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    processed_files = []

    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(image_files)}")

        cleaned_img = clean_and_standardize_image(img_path)

        if cleaned_img is not None:
            filename = Path(img_path).name
            output_path = os.path.join(output_dir, filename)

            img_to_save = (cleaned_img * 255).astype(np.uint8)

            # FIXED
            cv2.imwrite(output_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

            processed_files.append(output_path)

    return processed_files


all_cleaned_files = preprocess_all_images(all_image_files, CLEANED_DIR)
print(f"Total cleaned images: {len(all_cleaned_files)}")

# -----------------------------
# Create labels
# -----------------------------
print("Creating labels...")

def smart_label_extractor(file_path):
    filename = os.path.basename(file_path).lower()
    first_two = filename[:2]
    if first_two in ['cr','in','pa','pi','ro','sc']:
        return first_two
    return None

full_df = pd.DataFrame({'file': all_cleaned_files})
full_df['label_name'] = full_df['file'].map(smart_label_extractor)
full_df = full_df.dropna(subset=['label_name'])

unique_labels = sorted(full_df['label_name'].unique())
label_name_to_label_id = {label:i for i,label in enumerate(unique_labels)}

full_df['label'] = full_df['label_name'].map(label_name_to_label_id)

NUM_CLASSES = len(unique_labels)
print(f"Labels found: {unique_labels}")

# -----------------------------
# Split 70/15/15
# -----------------------------
train_df, temp_df = train_test_split(full_df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save CSVs
full_df.to_csv(CLEANED_DIR / 'full_cleaned_dataset.csv', index=False)
train_df.to_csv(CLEANED_DIR / 'train_dataset.csv', index=False)
val_df.to_csv(CLEANED_DIR / 'val_dataset.csv', index=False)
test_df.to_csv(CLEANED_DIR / 'test_dataset.csv', index=False)

print("CSV files saved.")

# -----------------------------
# DataLoaders
# -----------------------------
print("PHASE 2: Preparing DataLoaders...")

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file',
    y_col='label_name',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file',
    y_col='label_name',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='file',
    y_col='label_name',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# -----------------------------
# Train EfficientNetB0
# -----------------------------
print("PHASE 3: Training EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------------
# Fine-tuning
# -----------------------------
base_model.trainable = True
fine_tune_at = int(len(base_model.layers)*0.8)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1]
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Fine-tuned Test Accuracy: {test_acc:.4f}")

print("DONE!")