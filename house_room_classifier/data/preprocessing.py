import tensorflow as tf
import os
import hashlib
from collections import defaultdict

# Optimized load_dataset function with caching and prefetching
def load_dataset(data_dir, img_height=150, img_width=150, batch_size=20, subset=None, validation_split=0.2, seed=123, shuffle=True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=shuffle,
        validation_split=validation_split if subset else None,
        subset=subset,
        seed=seed
    )
    # Optimize dataset with caching and prefetching
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Optimized load_datasets function
def load_datasets(train_dir, val_dir=None, test_dir=None, img_height=150, img_width=150, batch_size=20, validation_split=0.2, seed=123):
    if val_dir and test_dir:
        # If separate directories are provided for all sets
        train_ds = load_dataset(train_dir, img_height, img_width, batch_size, seed=seed)
        val_ds = load_dataset(val_dir, img_height, img_width, batch_size, shuffle=False, seed=seed)
        test_ds = load_dataset(test_dir, img_height, img_width, batch_size, shuffle=False, seed=seed)
    else:
        # Split training directory into training and validation datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            validation_split=validation_split,
            subset='training',
            seed=seed
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            validation_split=validation_split,
            subset='validation',
            shuffle=False,
            seed=seed
        ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        test_ds = None

    return train_ds, val_ds, test_ds

# Apply augmentations to a dataset
def apply_augmentations(dataset, data_augmentation):
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Apply augmentations to a single image
def apply_augmentations_image(image, data_augmentation):
    if len(image.shape) == 3:  # Assuming image is (H, W, C)
        image = tf.expand_dims(image, axis=0)
    augmented_image = data_augmentation(image, training=True)
    return tf.squeeze(augmented_image, axis=0)

# Apply normalization to a dataset
def apply_normalization(dataset, normalization):
    return dataset.map(lambda x, y: (normalization(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE)

# Compute the hash of an image file
def compute_image_hash(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return hashlib.md5(image_bytes).hexdigest()

# Find and remove duplicate images
def find_and_remove_duplicates(dataset_path):
    hash_dict = defaultdict(list)
    duplicate_set = set()
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_hash = compute_image_hash(file_path)
                hash_dict[file_hash].append(file_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    for hash_value, paths in duplicates.items():
        duplicate_set.update(paths[1:])  # Keep the first occurrence, remove others
        for path in paths[1:]:
            os.remove(path)

    return duplicate_set, len(duplicate_set)

# Remove noise from an image (e.g., resizing and normalizing)
def remove_noise(image_path, target_size=(224, 224)):
    """Load and preprocess an image for the model."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values if needed
    return img_array