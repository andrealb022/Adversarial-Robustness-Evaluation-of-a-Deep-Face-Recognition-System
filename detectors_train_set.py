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

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2

# Funzione per creare il train set per i detectors a partire dal dataset originale (randomicamente).
def create_train_set_clean(dataset_directory_origin, dataset_directory_destination, number_img):
    # Creazione della lista contenente tutte le sottocartelle (classi) del dataset di partenza:
    class_folders = [os.path.join(dataset_directory_origin, d) for d in os.listdir(dataset_directory_origin) if os.path.isdir(os.path.join(dataset_directory_origin, d))]
    # Creazione della lista contenente tutte le immagini di tutte le sottocartelle:
    all_images = []
    for folder in class_folders:
        images = glob(os.path.join(folder, '*.jpg'))
        all_images.extend(images)
    # Verifica che ci sono abbastanza immagini:
    if number_img > len(all_images):
        raise ValueError(f"Numero di immagini richieste ({number_img}) maggiore del numero di immagini disponibili ({len(all_images)}).")
    # Selezione randomica del numero di immagini desiderate:
    selected_images = random.sample(all_images, number_img)
    # Creazione della directory di destinazione:
    os.makedirs(dataset_directory_destination, exist_ok=True)
    # Copia e rinomina delle immagini nella cartella di destinazione:
    for i, image_path in enumerate(selected_images):
        filename = f"img_{i:05d}.jpg"
        destination_path = os.path.join(dataset_directory_destination, filename)
        shutil.copy(image_path, destination_path)
    print(f"Copiate {len(selected_images)} immagini.")

# Funzione per ridimensionare le immagini clean in 224x224.
def process_clean_images(dataset_directory_destination, dataset_directory_processed):
    for root, _, files in os.walk(dataset_directory_destination):
        for fname in tqdm(files, desc="Processing"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = transforms.Resize(256)(image) # resize del lato corto dell'immagine a 256 (mantenendo la proporzione sul lato lungo)
                    image = transforms.CenterCrop(224)(image) # ritaglia il centro dell'immagine 224x224 
                    relative_path = os.path.relpath(img_path, dataset_directory_destination)
                    save_path = os.path.join(dataset_directory_processed, relative_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path)
                except Exception as e:
                    print(f"Errore all'immagine {img_path}: {e}.")

# Trasformazione da applicare alle immagini per convertire il range di valori da [0, 255] a [-1,1].
trans = transforms.Compose([
    transforms.ToTensor(), # converte l'immagine [0, 255] in tensore [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalizzazione per convertire da [0, 1] a [-1, 1] (su ogni canale)
])

def get_train_set():
    return TrainSet(images_dir="./dataset/detectors_train_set/clean/processed")

# Classe per la lettura e la gestione delle immagini e delle etichette del train set (clean) dei detectors.
class TrainSet(Dataset):
    def __init__(self, images_dir):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"La directory {images_dir} non esiste.")
        # Lettura della directory per creare una lista contenente i path di ogni immagine del train set
        self.samples = []
        for i, fname in enumerate(os.listdir(images_dir)):
            if fname.endswith('.jpg'):
                self.samples.append(os.path.join(images_dir, fname))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"Nessuna immagine .jpg trovata nella directory {images_dir}.")

    # Metodo che restituisce la dimensione del train set
    def __len__(self):
        return len(self.samples)

    # Metodo che restituisce l'immagine di un elemento specifico del train set
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path)
        return trans(image)
    
    # Metodo che restituisce tutte le immagine del train set
    def get_images(self):
        dataloader = DataLoader(self, batch_size=32, shuffle=False)
        train_images = []
        for images in dataloader:
            train_images.append(images.numpy())
        train_images = np.concatenate(train_images, axis=0)
        return train_images

# Funzione per creare il train set (adversarial) dei detectors (randomicamente).
def create_train_set_adv(classifier, images, attack_types):
    num_images = images.shape[0]
    untargeted_images = num_images // 2 
    targeted_images = num_images - untargeted_images
    for attack_name in attack_types:
        # Generazione campioni adversarial FGSM al variare di epsilon (50% targeted e 50% untargeted):
        if attack_name == "fgsm":
            attack = FGSM(classifier)
            all_adv_images = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/fgsm"
            for i in tqdm(range(untargeted_images), desc = "Generating untargeted adversarial examples"):
                eps = random.uniform(0.01, 0.1)
                all_adv_images.append(attack.generate_images(images[i:i+1], eps))
            for j in tqdm(range(targeted_images), desc = "Generating targeted adversarial examples"):
                targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv_images.append(attack.generate_images(images[untargeted_images+j:untargeted_images+j+1], eps, targeted=True, targeted_labels=targeted_labels))
            all_adv_images = np.concatenate(all_adv_images, axis=0)
            save_images_as_npy(all_adv_images, f"random_train_set", save_dir)
            print(f"Training adversarial examples generated and saved successfully for fgsm ({len(all_adv_images)} campioni).")
        # Generazione campioni adversarial BIM al variare di epsilon (50% targeted e 50% untargeted):
        if attack_name == "bim":
            attack = BIM(classifier)
            all_adv_images = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/bim"
            for i in tqdm(range(untargeted_images), desc = "Generating untargeted adversarial examples"):
                eps = random.uniform(0.01, 0.1)
                all_adv_images.append(attack.generate_images(images[i:i+1], eps, epsilon_step=0.01, max_iter=10))
            for j in tqdm(range(targeted_images), desc = "Generating targeted adversarial examples"):
                targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv_images.append(attack.generate_images(images[untargeted_images+j:untargeted_images+j+1], eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=targeted_labels))
            all_adv_images = np.concatenate(all_adv_images, axis=0)
            save_images_as_npy(all_adv_images, f"random_train_set", save_dir)
            print(f"Training adversarial examples generated and saved successfully for bim ({len(all_adv_images)} campioni).")
        # Generazione campioni adversarial PGD al variare di epsilon (50% targeted e 50% untargeted):
        if attack_name == "pgd":
            attack = PGD(classifier)
            all_adv_images = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/pgd"
            for i in tqdm(range(untargeted_images), desc = "Generating untargeted adversarial examples"):
                eps = random.uniform(0.01, 0.1)
                all_adv_images.append(attack.generate_images(images[i:i+1], eps, epsilon_step=0.01, max_iter=10))
            for j in tqdm(range(targeted_images), desc = "Generating targeted adversarial examples"):
                targeted_labels = torch.randint(low=0, high=NUM_CLASSES-1, size=(1,))
                all_adv_images.append(attack.generate_images(images[untargeted_images+j:untargeted_images+j+1], eps, epsilon_step=0.01, max_iter=10, targeted=True, targeted_labels=targeted_labels))
            all_adv_images = np.concatenate(all_adv_images, axis=0)
            save_images_as_npy(all_adv_images, f"random_train_set", save_dir)
            print(f"Training adversarial examples generated and saved successfully for pgd ({len(all_adv_images)} campioni).")
        # Generazione campioni adversarial DF al variare di epsilon (100% untargeted perché la versione targeted di DF non è supportata):
        if attack_name == "df":
            attack = DF(classifier)
            all_adv_images = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/df"
            for i in tqdm(range(num_images), desc = "Generating untargeted adversarial examples"):
                eps = random.uniform(1e-5, 1.0)
                all_adv_images.append(attack.generate_images(images[i:i+1], eps, nb_grads=10, max_iter=10))
            all_adv_images = np.concatenate(all_adv_images, axis=0)
            save_images_as_npy(all_adv_images, f"random_train_set", save_dir)
            print(f"Training adversarial examples generated and saved successfully for df ({len(all_adv_images)} campioni).")
        # Generazione campioni adversarial CW al variare di confidence (100% untargeted perché la versione targeted di CW non è efficace):
        if attack_name == "cw":
            attack = CW(classifier)
            all_adv_images = []
            save_dir = "./dataset/detectors_train_set/adversarial_examples/cw"
            for i in tqdm(range(num_images), desc = "Generating untargeted adversarial examples"):
                conf = random.uniform(0.01, 1.0)
                all_adv_images.append(attack.generate_images(images[i:i+1], conf, learning_rate=0.01, max_iter=3))
            all_adv_images = np.concatenate(all_adv_images, axis=0)
            save_images_as_npy(all_adv_images, f"random_train_set", save_dir)
            print(f"Training adversarial examples generated and saved successfully for cw ({len(all_adv_images)} campioni).")

if __name__ == "__main__":
    random.seed(33)
    
    # Generazione del train set clean
    dataset_directory_origin = './dataset/vggface2_train/train' # Directory contenente il dataset originale
    dataset_directory_destination = './dataset/detectors_train_set/clean/original' # Directory in cui salvare le immagini selezionate
    dataset_directory_processed = './dataset/detectors_train_set/clean/processed' # Directory in cui salvare le immagini proccessate
    dim_train_set = 1000 # numero di immagini del train set per i detectors
    create_train_set_clean(dataset_directory_origin, dataset_directory_destination, dim_train_set) # crea il train set clean
    process_clean_images(dataset_directory_destination, dataset_directory_processed) # processa le immagini del train set clean
    
    # Generazione del train set adversarial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = setup_NN1_classifier(device)
    train_images_clean = get_train_set().get_images()
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]
    create_train_set_adv(classifier, train_images_clean, attack_types) # crea il train set adversarial