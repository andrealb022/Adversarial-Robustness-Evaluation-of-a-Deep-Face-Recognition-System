import os
import csv
import random
import shutil
import tqdm
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Function to create the test set from the original dataset and a CSV file containing selected identities
def create_test_set(csv_file, original_dataset_dir, destination_dataset_dir, num_images_per_identity):
    # Create the destination directory
    os.makedirs(destination_dataset_dir, exist_ok=True)
    
    # Read the CSV file containing selected identity IDs
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            person_id = row[0].strip()
            person_name = row[1].strip(' "')
            
            # Build source and destination paths for each identity
            origin_path = os.path.join(original_dataset_dir, person_id)
            dest_path = os.path.join(destination_dataset_dir, person_id)
            if not os.path.exists(origin_path):
                print(f"Directory {origin_path} not found.")
                continue

            # List images in the identity's folder
            images = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
            if not images:
                print(f"No images found in directory {origin_path}.")
                continue

            # Randomly select the desired number of images
            selected_images = random.sample(images, min(num_images_per_identity, len(images)))
            
            # Create the destination directory
            os.makedirs(dest_path, exist_ok=True)

            # Copy and rename images to the destination folder
            for idx, image in enumerate(selected_images):
                src = os.path.join(origin_path, image)
                ext = os.path.splitext(image)[1]
                dst_filename = f"{person_id}_{idx + 1:02d}{ext}"
                dst = os.path.join(dest_path, dst_filename)
                shutil.copy2(src, dst)
            
            print(f"Copied {len(selected_images)} images for {person_name}.")

# Function to resize clean images to 224x224
def process_clean_images(src_dir, processed_dir):
    for root, _, files in os.walk(src_dir):
        for fname in tqdm.tqdm(files, desc="Processing"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transforms.Resize(256)(image)  # resize the short side to 256 (preserve aspect ratio)
                    image = transforms.CenterCrop(224)(image)  # crop the center to 224x224
                    relative_path = os.path.relpath(img_path, src_dir)
                    save_path = os.path.join(processed_dir, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}.")

# Transformation to convert image value range from [0, 255] to [-1, 1]
image_transform = transforms.Compose([
    transforms.ToTensor(),  # convert [0, 255] to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1, 1]
])

def get_test_set():
    return TestSet(
        images_dir="./dataset/test_set/clean/processed",
        csv_path="./dataset/test_set/test_set.csv",
        label_map_path="./dataset/rcmalli_vggface_labels_v2.npy"
    )

# Dataset class for loading test images and labels
class TestSet(Dataset):
    def __init__(self, images_dir, csv_path, label_map_path):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Directory {images_dir} does not exist.")
        
        # Load identity labels from a .npy file mapping identity names to integers
        LABELS = np.load(label_map_path)
        self.true_labels = {str(name).strip(): idx for idx, name in enumerate(LABELS)}

        # Read the CSV to create (image_path, label) pairs
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                person_dir, name = row[0], row[1].strip(' "')
                full_dir = os.path.join(images_dir, person_dir)
                if os.path.isdir(full_dir):
                    for _, img_file in enumerate(os.listdir(full_dir)):
                        img_path = os.path.join(full_dir, img_file)
                        if os.path.isfile(img_path):
                            label = self.true_labels.get(name)
                            if label is not None:
                                self.samples.append((img_path, label))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in directory {self.images_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        return image_transform(image), label

    def get_true_label(self, name):
        return self.true_labels.get(name)

    def get_used_labels(self):
        return sorted({label for _, label in self.samples})

    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        test_images, test_labels = [], []
        for images, labels in dataloader:
            test_images.append(images.numpy())
            test_labels.append(labels)
        test_images = np.concatenate(test_images, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        return test_images, test_labels


if __name__ == "__main__":
    random.seed(33)

    # Test set generation (clean)
    csv_file = './dataset/test_set/test_set.csv'  # CSV file listing selected identities
    original_dataset_dir = './dataset/vggface2_train/train'  # Path to the original dataset
    destination_dataset_dir = './dataset/test_set/clean/original'  # Path to save selected images
    processed_dataset_dir = './dataset/test_set/clean/processed'  # Path to save resized images
    num_images_per_identity = 10  # 10 images per identity (e.g., 10 x 100 = 1000 images total)

    create_test_set(csv_file, original_dataset_dir, destination_dataset_dir, num_images_per_identity)
    process_clean_images(destination_dataset_dir, processed_dataset_dir)