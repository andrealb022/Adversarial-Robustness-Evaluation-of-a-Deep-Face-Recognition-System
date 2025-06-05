import os
import numpy as np
import torch
from nets import setup_detector_classifier
from art.defences.detector.evasion import BinaryInputDetector

# Function to save a numpy array of images to a '.npy' file in a specified directory
def save_images_as_npy(images, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist
    filepath = os.path.join(save_dir, f"{filename}.npy")
    np.save(filepath, images)  # Save the image array to file

# Function to load all '.npy' image arrays from a specified folder
def load_images_from_npy(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")
    images_list = []
    for file_name in sorted(files):  # Ensure files are loaded in consistent order
        file_path = os.path.join(folder_path, file_name)
        images_array = np.load(file_path)
        images_list.append(images_array)
    return images_list

# Function to load pretrained detector models for various attack types from disk
def load_detectors(attack_types, device):
    detectors = {}
    for attack_type in attack_types:
        model_path = os.path.join("./models", f"{attack_type}_detector.pth")  # Path to the saved detector
        detector_classifier = setup_detector_classifier(device)  # Create the classifier model
        # Load model weights from file
        detector_classifier.model.load_state_dict(torch.load(model_path, map_location=device))
        detector_classifier.model.eval()  # Set model to evaluation mode
        # Wrap the classifier with BinaryInputDetector from ART
        detectors[attack_type] = BinaryInputDetector(detector_classifier)
        print(f"Loaded detector from: {model_path}")
    return detectors

# Function to preprocess images before passing them to the second-stage network (NN2)
def process_images(images):
    processed_images = []  # List to hold processed images
    # Mean pixel values for B, G, R channels (as used during NN2 training on VGGFace2)
    mean_bgr = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)
    
    for image in images:
        # Convert from [-1, 1] float32 to [0, 255] float32 (assuming original normalization was [-1, 1])
        image = (image + 1.0) * (255.0 / 2)
        # Convert from RGB to BGR channel order
        image = image[[2, 1, 0], :, :]
        # Subtract mean for each channel (as required by NN2)
        image -= mean_bgr
        processed_images.append(image)
        
    return np.stack(processed_images, axis=0)  # Return batch as a numpy array