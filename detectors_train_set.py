import os
import random
import shutil
import numpy as np
import torch
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from attacks import BIM, CW, DF, FGSM, PGD
from nets import setup_NN1_classifier
from utils import save_images_as_npy

NUM_CLASSES = 8631  # number of classes in the VGGFace2 dataset

# Function to create the training set (clean) for detectors by randomly sampling from the original dataset.
def create_clean_train_set(origin_dir, dest_dir, num_images):
    # Get list of all class subfolders in the origin dataset directory:
    class_folders = [os.path.join(origin_dir, d) for d in os.listdir(origin_dir) if os.path.isdir(os.path.join(origin_dir, d))]
    # Get list of all images in all subfolders:
    all_images = []
    for folder in class_folders:
        images = glob(os.path.join(folder, '*.jpg'))
        all_images.extend(images)
    # Check that there are enough images:
    if num_images > len(all_images):
        raise ValueError(f"Requested number of images ({num_images}) exceeds available images ({len(all_images)}).")
    # Randomly select the desired number of images:
    selected_images = random.sample(all_images, num_images)
    # Create destination directory:
    os.makedirs(dest_dir, exist_ok=True)
    # Copy and rename images into the destination directory:
    for i, image_path in enumerate(selected_images):
        filename = f"img_{i:05d}.jpg"
        destination_path = os.path.join(dest_dir, filename)
        shutil.copy(image_path, destination_path)
    print(f"Copied {len(selected_images)} images.")

# Function to resize clean images to 224x224.
def preprocess_clean_images(src_dir, processed_dir):
    for root, _, files in os.walk(src_dir):
        for fname in tqdm(files, desc="Processing"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transforms.Resize(256)(image)  # resize shortest side to 256 keeping aspect ratio
                    image = transforms.CenterCrop(224)(image)  # crop to center 224x224
                    relative_path = os.path.relpath(img_path, src_dir)
                    save_path = os.path.join(processed_dir, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)
                except Exception as e:
                    print(f"Error with image {img_path}: {e}")

# Transformation to normalize images from [0, 255] → [-1, 1]
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_clean_train_set():
    return CleanTrainSet(images_dir="./dataset/detectors_train_set/clean/processed")

# Dataset class to load and manage clean images for detectors' training set.
class CleanTrainSet(Dataset):
    def __init__(self, images_dir):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Directory {images_dir} does not exist.")
        self.samples = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No .jpg images found in {images_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path)
        return transform_pipeline(image)

    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        all_images = []
        for images in dataloader:
            all_images.append(images.numpy())
        return np.concatenate(all_images, axis=0)

# Function to generate adversarial training set for detectors.
def create_adv_train_set(classifier, images, attack_types):
    num_images = images.shape[0]
    untargeted_count = num_images // 2
    targeted_count = num_images - untargeted_count

    for attack_name in attack_types:
        if attack_name == "fgsm":
            attack = FGSM(classifier)
            all_adv = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/fgsm"
            for i in tqdm(range(untargeted_count), desc="Generating untargeted FGSM"):
                eps = random.uniform(0.01, 0.1)
                all_adv.append(attack.generate_images(images[i:i+1], eps))
            for j in tqdm(range(targeted_count), desc="Generating targeted FGSM"):
                labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv.append(attack.generate_images(images[untargeted_count+j:untargeted_count+j+1], eps, targeted=True, targeted_labels=labels))
            all_adv = np.concatenate(all_adv, axis=0)
            save_images_as_npy(all_adv, "random_train_set", save_dir)
            print(f"FGSM adversarial training examples generated: {len(all_adv)}")

        if attack_name == "bim":
            attack = BIM(classifier)
            all_adv = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/bim"
            for i in tqdm(range(untargeted_count), desc="Generating untargeted BIM"):
                eps = random.uniform(0.01, 0.1)
                all_adv.append(attack.generate_images(images[i:i+1], eps, epsilon_step=0.01, max_iter=10))
            for j in tqdm(range(targeted_count), desc="Generating targeted BIM"):
                labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv.append(attack.generate_images(images[untargeted_count+j:untargeted_count+j+1], eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=labels))
            all_adv = np.concatenate(all_adv, axis=0)
            save_images_as_npy(all_adv, "random_train_set", save_dir)
            print(f"BIM adversarial training examples generated: {len(all_adv)}")

        if attack_name == "pgd":
            attack = PGD(classifier)
            all_adv = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/pgd"
            for i in tqdm(range(untargeted_count), desc="Generating untargeted PGD"):
                eps = random.uniform(0.01, 0.1)
                all_adv.append(attack.generate_images(images[i:i+1], eps, epsilon_step=0.01, max_iter=10))
            for j in tqdm(range(targeted_count), desc="Generating targeted PGD"):
                labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv.append(attack.generate_images(images[untargeted_count+j:untargeted_count+j+1], eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=labels))
            all_adv = np.concatenate(all_adv, axis=0)
            save_images_as_npy(all_adv, "random_train_set", save_dir)
            print(f"PGD adversarial training examples generated: {len(all_adv)}")

        if attack_name == "df":
            attack = DF(classifier)
            all_adv = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/df"
            for i in tqdm(range(num_images), desc="Generating untargeted DF"):
                eps = random.uniform(1e-5, 1.0)
                all_adv.append(attack.generate_images(images[i:i+1], eps, nb_grads=10, max_iter=10))
            all_adv = np.concatenate(all_adv, axis=0)
            save_images_as_npy(all_adv, "random_train_set", save_dir)
            print(f"DF adversarial training examples generated: {len(all_adv)}")

        if attack_name == "cw":
            attack = CW(classifier)
            all_adv = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/cw"
            for i in tqdm(range(num_images), desc="Generating untargeted CW"):
                conf = random.uniform(0.01, 1.0)
                all_adv.append(attack.generate_images(images[i:i+1], conf, learning_rate=0.01, max_iter=3))
            all_adv = np.concatenate(all_adv, axis=0)
            save_images_as_npy(all_adv, "random_train_set", save_dir)
            print(f"CW adversarial training examples generated: {len(all_adv)}")

if __name__ == "__main__":
    random.seed(33)

    # Create clean training set
    origin_dir = './dataset/vggface2_train/train'
    clean_original_dir = './dataset/detectors_train_set/clean/original'
    clean_processed_dir = './dataset/detectors_train_set/clean/processed'
    train_set_size = 1000
    create_clean_train_set(origin_dir, clean_original_dir, train_set_size)
    preprocess_clean_images(clean_original_dir, clean_processed_dir)

    # Create adversarial training set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = setup_NN1_classifier(device)
    clean_images = get_clean_train_set().get_images()
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    create_adv_train_set(classifier, clean_images, attack_types)