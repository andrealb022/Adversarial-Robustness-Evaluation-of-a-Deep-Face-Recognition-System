import numpy as np
import os
import torch
from detectors_train_set import get_train_set
from nets import setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector
from utils import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Crezione della directory in cui salvare i modelli dei detectors:
    os.makedirs("./models", exist_ok=True)
    
    # Lettura del train set (clean) dei detectors:
    train_images_clean = get_train_set().get_images()

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Addestramento dei detectors (uno per ogni attacco):
    for attack_type in attack_types:
        print(f"Training detector for attack: {attack_type}")

        # Creazione di un'istanza della classe BinaryInputDetector della libreria ART:
        detector_classifier = setup_detector_classifier(device)
        detector = BinaryInputDetector(detector_classifier)
        
        # Lettura del train set (adversarial) dei detectors:
        training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples/", attack_type)
        train_images_adv = load_images_from_npy(training_set_path)
        train_images_adv = np.concatenate(train_images_adv, axis=0)
        
        # Concatenazione delle immagini clean e adversarial:
        x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

        # Creazione delle etichette per il training set ([1, 0] per immagini clean e [0, 1] per immagini adversarial):
        y_train_detector = np.concatenate((np.array([[1, 0]] * train_images_clean.shape[0]), np.array([[0, 1]] * np.shape(train_images_adv)[0])), axis=0)

        # Addestramento del detector:
        detector.fit(x_train_detector, y_train_detector, nb_epochs=30, batch_size=16, verbose=True)
        
        # Salvataggio del modello del detector addestrato:
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier.model.eval()
        torch.save(detector_classifier.model.state_dict(), model_path)
        print(f"Detector salvato in: {model_path}.")

if __name__ == "__main__":
    main()