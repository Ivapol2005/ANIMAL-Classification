import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

# from breeds_list import breeds, cat_breeds, dog_breeds
from code.breeds_list import *

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_and_preprocess(image_path, label_breed, label_animal):
    image_path = image_path.numpy().decode("utf-8")
    image = cv2.imread(image_path)
    
    if image is None:
        return tf.zeros((IMG_SIZE[0], IMG_SIZE[1], 3)), tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    
    return image, label_breed, label_animal

def tf_load_and_preprocess(image_path, label_breed, label_animal):
    image, label_breed, label_animal = tf.py_function(load_and_preprocess, [image_path, label_breed, label_animal], [tf.float32, tf.int32, tf.int32])
    
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    label_breed.set_shape([])
    label_animal.set_shape([])

    return image, {"breed_output": label_breed, "animal_output": label_animal}

def create_tf_dataset(dataset):
    image_paths = [entry["image"] for entry in dataset]
    labels_breed = [entry["breed"] for entry in dataset]  # Порода
    labels_animal = [1 if entry["breed"] in dog_breeds else 0 for entry in dataset]  # 1 = собака, 0 = кіт

    image_paths = tf.constant(image_paths, dtype=tf.string)
    labels_breed = tf.constant(labels_breed, dtype=tf.int32)
    labels_animal = tf.constant(labels_animal, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_breed, labels_animal))
    ds = ds.map(tf_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return ds
    

def parse_voc_annotation(xml_file):
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})

    return objects


def load_voc_dataset(images_dir, annotations_dir):
    dataset = []
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".xml"):
            image_file = os.path.join(images_dir, filename.replace(".xml", ".jpg"))
            annotation_file = os.path.join(annotations_dir, filename)
            
            image_name = filename.replace(".xml", "")
            image_data = df.loc[df["Image"] == image_name]

            if os.path.exists(image_file):    
                annotations = parse_voc_annotation(annotation_file)
                
                if not image_data.empty:
                    species_id = int(image_data["Species"].values[0])
                    breed_id = int(image_data["Breed_ID"].values[0])
                    
                    dataset.append({'image': image_file,
                                    'name': image_name,
                                    'animal': {1: "Cat", 2: "Dog"}.get(species_id),
                                    'breed': breed_id, # breeds.get(breed_id),
                                    'annotations': annotations
                                   })
                else:
                    dataset.append({'image': image_file,
                                    'name': image_name,
                                    'animal': 0,
                                    'breed': 0,
                                    'annotations': annotations
                                   })

    return dataset
    
    

def load_voc_dataset_test(images_dir, annotations_file):
    df = pd.read_csv(annotations_file, comment="#", sep=" ", header=None, 
                     names=["Image", "Class_ID", "Species", "Breed_ID"])
    
    test_images = set(df_test["Image"].tolist())
    
    dataset = []
    
    for image_name in test_images:
        image_file = os.path.join(images_dir, image_name + ".jpg")
            
        image_data = df[df["Image"] == image_name]
            
        if not image_data.empty:
            species_id = int(image_data["Species"].values[0])
            breed_id = int(image_data["Breed_ID"].values[0])
                
            dataset.append({
                'image': image_file,
                'name': image_name,
                'animal': {1: "Cat", 2: "Dog"}.get(species_id),
                'breed': breed_id, # breeds.get(breed_id, "Unknown Breed"),
                'annotations': []  # No XML, so annotations are empty
            })
        else:
            dataset.append({
                'image': image_file,
                'name': image_name,
                'animal': 0,
                'breed': 0,
                'annotations': []  # No XML, so annotations are empty
            })
    
    return dataset


import pandas as pd

file_path = "dataset/annotations/list.txt"
df = pd.read_csv(file_path, comment="#", sep=" ", header=None, names=["Image", "Class_ID", "Species", "Breed_ID"])

images_dir = "dataset/images"
annotations_dir = "dataset/annotations/xmls"

dataset = load_voc_dataset(images_dir, annotations_dir)
# print(f"Завантажено {len(dataset)} зображень.")
# print("Приклад:", dataset[0])

file_test_path = "dataset/annotations/test.txt"
df_test = pd.read_csv(file_test_path, comment="#", sep=" ", header=None, names=["Image", "Class_ID", "Species", "Breed_ID"])

dataset_test = load_voc_dataset_test(images_dir, file_test_path)
dataset_train = dataset

# Create TensorFlow Datasets
train_ds = create_tf_dataset(dataset)
test_ds = create_tf_dataset(dataset_test)
