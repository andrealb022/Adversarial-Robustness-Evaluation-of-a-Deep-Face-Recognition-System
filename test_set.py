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

# Funzione per creare il test set a partire dal dataset originale e da un file CSV contenente le identità scelte.
def create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img_for_person):
    # Creazione della directory di destinazione:
    os.makedirs(dataset_directory_destination, exist_ok=True)
    # Lettura del file CSV contenente gli ID delle identità scelte:
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            person_id = row[0].strip()
            person_name = row[1].strip(' "')
            # Costruzione dei percorsi di origine e destinazione per ogni identità:
            origin_path = os.path.join(dataset_directory_origin, person_id)
            destination_path = os.path.join(dataset_directory_destination, person_id)
            if not os.path.exists(origin_path):
                print(f"Cartella {origin_path} non trovata.")
                continue
            # Creazione della lista di immagini presenti nella cartella dell'identità:
            images = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
            if not images:
                print(f"Nessuna immagine nella cartella {origin_path}.")
                continue
            # Selezione randomica del numero di immagini desiderate:
            selected_images = random.sample(images, min(number_img_for_person, len(images)))
            # Creazione la directory di destinazione:
            os.makedirs(destination_path, exist_ok=True)
            # Copia e rinomina delle immagini nella cartella di destinazione:
            for idx, image in enumerate(selected_images):
                src = os.path.join(origin_path, image)
                ext = os.path.splitext(image)[1]
                dst_filename = f"{person_id}_{idx + 1:02d}{ext}"
                dst = os.path.join(destination_path, dst_filename)
                shutil.copy2(src, dst)
            print(f"Copiate {len(selected_images)} immagini per {person_name}.")

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

def get_test_set():
    return TestSet(images_dir="./dataset/test_set/clean/processed",
                   csv_path="./dataset/test_set/test_set.csv",
                   label_map_path="./dataset/rcmalli_vggface_labels_v2.npy")

# Classe per la lettura e la gestione delle immagini e delle etichette del test set.
class TestSet(Dataset):
    def __init__(self, images_dir, csv_path, label_map_path):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"La directory {images_dir} non esiste.")
        
        # Caricamento delle true lables in un dizionario che mappa il nome di un'identità con un numero intero:
        LABELS = np.load(label_map_path)
        self.true_labels = {str(name).strip(): idx for idx, name in enumerate(LABELS)}
        
        # Lettura del file CSV del test set per creare una lista contenente (img_path, label) per ogni immagine del test set
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                person_dir, name = row[0], row[1].strip(' "')
                full_dir = os.path.join(images_dir, person_dir)
                if os.path.isdir(full_dir):
                    for i, img_file in enumerate(os.listdir(full_dir)):
                        img_path = os.path.join(full_dir, img_file)
                        if os.path.isfile(img_path):
                            label = self.true_labels.get(name)
                            if label is not None:
                                self.samples.append((img_path, label))
        if len(self.samples) == 0:
            raise FileNotFoundError(f"Nessuna immagine trovata nella directory {self.images_dir}.")

    # Metodo che restituisce la dimensione del test set
    def __len__(self):
        return len(self.samples)

    # Metodo che restituisce l'immagine e l'etichetta di un elemento specifico del test set
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        return trans(image), label

    # Metodo che restituisce l'etichetta di un identità
    def get_true_label(self, name):
        return self.true_labels.get(name)

    # Metodo che restituisce l'etichette delle identità usate per il test set
    def get_used_labels(self):
        return sorted({label for _, label in self.samples})
    
    # Metodo che restituisce tutte le immagine con le rispettive etichette del test set
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

    # Generazione del test set (clean)
    csv_file = './dataset/test_set/test_set.csv' # File CSV contenente le identità da inserire nel test set
    dataset_directory_origin = './dataset/vggface2_train/train' # Directory contenente il dataset originale
    dataset_directory_destination = './dataset/test_set/clean/original' # Directory in cui salvare le immagini selezionate
    dataset_directory_processed = './dataset/test_set/clean/processed' # Directory in cui salvare le immagini proccessate
    number_img_for_person = 10 # Numero di immagini da copiare per ogni identità (10 per 100 persone = 1000 img)
    create_test_set(csv_file, dataset_directory_origin, dataset_directory_destination, number_img_for_person) # crea il test set
    process_clean_images(dataset_directory_destination, dataset_directory_processed) # processa le immagini del test set