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

    # Create directory to save detector models
    os.makedirs("./models", exist_ok=True)
    
    # Load the clean training set for detectors
    train_images_clean = get_train_set().get_images()

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Train one detector for each attack
    for attack_type in attack_types:
        print(f"Training detector for attack: {attack_type}")

        # Create an instance of ART's BinaryInputDetector
        detector_classifier = setup_detector_classifier(device)
        detector = BinaryInputDetector(detector_classifier)
        
        # Load adversarial training set for the current attack
        training_set_path = os.path.join("./dataset/detectors_train_set/adversarial_examples/", attack_type)
        train_images_adv = load_images_from_npy(training_set_path)
        train_images_adv = np.concatenate(train_images_adv, axis=0)
        
        # Concatenate clean and adversarial images
        x_train_detector = np.concatenate((train_images_clean, train_images_adv), axis=0)

        # Create training labels ([1, 0] for clean, [0, 1] for adversarial)
        y_train_detector = np.concatenate(
            (np.array([[1, 0]] * train_images_clean.shape[0]), 
             np.array([[0, 1]] * train_images_adv.shape[0])),
            axis=0
        )

        # Train the detector
        detector.fit(x_train_detector, y_train_detector, nb_epochs=30, batch_size=16, verbose=True)
        
        # Save the trained detector model
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")
        detector_classifier.model.eval()
        torch.save(detector_classifier.model.state_dict(), model_path)
        print(f"Detector saved at: {model_path}.")

if __name__ == "__main__":
    main()