import os
import pandas as pd
import cv2

import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow.keras import layers

from breeds_list import breeds, cat_breeds, dog_breeds


file_path = "../../dataset/annotations/list.txt"
file_test_path = "../../dataset/annotations/test.txt"

images_dir = "../../dataset/images"
annotations_dir = "../../dataset/annotations/xmls"


df = pd.read_csv(file_path, comment="#", sep=" ", header=None,
    names=["Image", "Class_ID", "Species", "Breed_ID"])
df_test = pd.read_csv(file_test_path, comment="#", sep=" ", header=None,
    names=["Image", "Class_ID", "Species", "Breed_ID"])


# Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    size_elem = root.find('size')
    img_width = int(size_elem.find('width').text)
    img_height = int(size_elem.find('height').text)

    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})
    
    return objects, (img_width, img_height)


def get_true_info_from_filename(filename):
    filename_lower = filename.lower()

    for breed_id, breed_name in breeds.items():
        cleaned_breed_name = breed_name.replace('_cat', '').replace('_dog', '').lower()
        if cleaned_breed_name in filename_lower:
            true_animal_type = None
            if breed_id in cat_breeds:
                true_animal_type = "Cat"
            elif breed_id in dog_breeds:
                true_animal_type = "Dog"

            if true_animal_type:
                return true_animal_type, breed_name

    return None, None


def load_dataset(images_dir, metadata_df, annotations_dir=None, is_test_set=False):
    dataset = []
    if annotations_dir and not is_test_set:
        image_names_to_process = [f.replace(".xml", "") for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    else:
        image_names_to_process = metadata_df["Image"].tolist()

    image_names_to_process = sorted(list(set(image_names_to_process)))

    for image_name_base in image_names_to_process:
        image_file = os.path.join(images_dir, image_name_base + ".jpg")

        image_data = metadata_df.loc[metadata_df["Image"] == image_name_base]

        if not os.path.exists(image_file):
            print(f"Warning: Image file not found for '{image_name_base}'. Skipping.")
            continue

        annotations_data = []
        original_img_dims = (0, 0)
        if annotations_dir and not is_test_set:
            annotation_file = os.path.join(annotations_dir, image_name_base + ".xml")
            if os.path.exists(annotation_file):
                annotations_data, original_img_dims = parse_voc_annotation(annotation_file)
            else:
                print(f"Warning: XML annotation file not found for '{image_name_base}'. Proceeding without annotations.")

        species_id = 0
        breed_id = 0
        animal_name = "Unknown"

        if not image_data.empty:
            detected_animal, detected_breed_name = get_true_info_from_filename(image_name_base)
            if detected_animal and detected_breed_name:
                animal_name = detected_animal
                for b_id, b_name in breeds.items():
                    if b_name == detected_breed_name:
                        breed_id = b_id

                        species_id = image_data['Species'].iloc[0]
                        break
            else:
                if is_test_set:
                    print(f"Warning: No metadata found in provided DataFrame for test image '{image_name_base}'. Assigning default labels.")
                else:
                    print(f"Warning: No metadata found in provided DataFrame for train image '{image_name_base}'. Assigning default labels.")
        

        if annotations_data:

            first_bbox = annotations_data[0]['bbox']
            orig_w, orig_h = original_img_dims
            if orig_w > 0 and orig_h > 0:
                normalized_bbox = (
                    first_bbox[0] / orig_w,
                    first_bbox[1] / orig_h,
                    first_bbox[2] / orig_w,
                    first_bbox[3] / orig_h
                )
            else:

                normalized_bbox = (0.0, 0.0, 0.0, 0.0)
        else:
            normalized_bbox = (0.0, 0.0, 0.0, 0.0)

        dataset.append({
            'image_path': image_file,
            'name': image_name_base,
            'animal': animal_name,
            'breed': breed_id,
            'species_id': species_id,
            'annotations': annotations_data,
            'bbox': normalized_bbox
        })

    return dataset


def load_and_preprocess(image_path, label_breed, label_animal, bbox_coords):
    image_path = image_path.numpy().decode("utf-8")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image {image_path}. Returning zeros.")
        return tf.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32), \
               tf.constant(0, dtype=tf.int32), \
               tf.constant(0, dtype=tf.int32), \
               tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0

    return image, label_breed, label_animal, bbox_coords


def tf_load_and_preprocess(image_path_tf, label_breed_tf, label_animal_tf, bbox_coords_tf):
    image, label_breed_id, label_animal_id, bbox = tf.py_function(
        load_and_preprocess,
        [image_path_tf, label_breed_tf, label_animal_tf, bbox_coords_tf],
        [tf.float32, tf.int32, tf.int32, tf.float32]
    )

    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label_breed_id.set_shape([])
    label_animal_id.set_shape([])
    bbox.set_shape([4])

    label_breed_one_hot = tf.one_hot(label_breed_id, depth=len(breeds))

    return image, {
        "breed_output": label_breed_one_hot,
        "animal_output": label_animal_id,
        "bbox_output": bbox
    }


def create_tf_dataset(dataset_list, augment=False):
    image_paths = [entry["image_path"] for entry in dataset_list]
    labels_breed = [entry["breed"] for entry in dataset_list]
    labels_animal = [0 if entry["species_id"] == 1 else 1 for entry in dataset_list]
    bboxes = [list(entry["bbox"]) for entry in dataset_list]

    image_paths_tf = tf.constant(image_paths, dtype=tf.string)
    labels_breed_tf = tf.constant(labels_breed, dtype=tf.int32)
    labels_animal_tf = tf.constant(labels_animal, dtype=tf.int32)
    bboxes_tf = tf.constant(bboxes, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths_tf, labels_breed_tf, labels_animal_tf, bboxes_tf))

    ds = ds.map(tf_load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        def apply_augmentation(image, labels):
            label_breed = labels["breed_output"]
            label_animal = labels["animal_output"]
            bbox = labels["bbox_output"]
            image = data_augmentation(image, training=True)

            label_breed = tf.reshape(label_breed, (len(breeds),))
            
            bbox = tf.reshape(bbox, (4,))

            return image, {
                "breed_output": label_breed,
                "animal_output": label_animal,
                "bbox_output": bbox
            }
        ds = ds.map(apply_augmentation, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds


# Load datasets
dataset_train = load_dataset(images_dir, df, annotations_dir=annotations_dir, is_test_set=False)
dataset_test = load_dataset(images_dir, df_test, annotations_dir=None, is_test_set=True)

train_ds = create_tf_dataset(dataset_train, augment=True).repeat()
test_ds = create_tf_dataset(dataset_test, augment=False).repeat()


# --- Annotation Printing Section (for verification) ---
print("Training Dataset (with BBox):")
for i, entry in enumerate(dataset_train):
    breed_name = breeds.get(entry['breed'], "Unknown Breed")

    animal_display = "Cat" if entry['species_id'] == 1 else "Dog" if entry['species_id'] == 2 else "Unknown"

    print(f"  {i+1}. Image: {entry['name']}.jpg, Animal: {animal_display}, Breed: {breed_name}")

    if entry['bbox'] and entry['bbox'] != (0.0, 0.0, 0.0, 0.0):
        print(f"    Normalized Bounding Box: {entry['bbox']}")
    else:
        print("    No Bounding Box information (or placeholder bbox).")

    if 'annotations' in entry and entry['annotations']:
        print("    Original Annotations:")
        for j, annotation in enumerate(entry['annotations']):
            box_coords = annotation.get('bbox', 'N/A')
            label = annotation.get('name', 'N/A')
            print(f"      - Annotation {j+1}: Box={box_coords}, Label='{label}'")
    elif 'annotations' in entry and not entry['annotations']:
        print("    No annotations for this image.")

print()

print("Test Dataset (with BBox):")
for i, entry in enumerate(dataset_test):
    breed_name = breeds.get(entry['breed'], "Unknown Breed")
    animal_display = "Cat" if entry['species_id'] == 1 else "Dog" if entry['species_id'] == 2 else "Unknown"
    print(f"  {i+1}. Image: {entry['name']}.jpg, Animal: {animal_display}, Breed: {breed_name}")

    if entry['bbox'] and entry['bbox'] != (0.0, 0.0, 0.0, 0.0):
        print(f"    Normalized Bounding Box: {entry['bbox']}")
    else:
        print("    No Bounding Box information (or placeholder bbox).")

    if 'annotations' in entry and entry['annotations']:
        print("    Original Annotations (unlikely for test set):")
        for j, annotation in enumerate(entry['annotations']):
            box_coords = annotation.get('bbox', 'N/A')
            label = annotation.get('name', 'N/A')
            print(f"      - Annotation {j+1}: Box={box_coords}, Label='{label}'")
    elif 'annotations' in entry and not entry['annotations']:
        print("    No annotations for this image.")

print("\nDataset inspection complete.")