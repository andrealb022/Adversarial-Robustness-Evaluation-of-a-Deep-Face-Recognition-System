import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from utils import *
from sklearn.metrics import *
from test_set import get_test_set

def get_adversarial_images(images_dir, num_samples):
    # Collect all .npy files in the given directory (and subfolders)
    all_npy_files = []
    files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
    npy_files = [os.path.join(images_dir, f) for f in sorted(files)]
    all_npy_files.extend(npy_files)

    # Calculate how many samples to take from each file
    samples_per_file = num_samples // len(all_npy_files)
    remainder = num_samples % len(all_npy_files)

    # Randomly sample images from each .npy file
    imgs_subset = []
    for i, npy_file in enumerate(all_npy_files):
        data = np.load(npy_file)
        n_samples = samples_per_file + (1 if i < remainder else 0)  # distribute remaining samples
        indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
        imgs_subset.append(data[indices])
    
    # Concatenate subsets of images
    imgs = np.concatenate(imgs_subset, axis=0).reshape(-1, 3, 224, 224)
    return imgs

def main():
    np.random.seed(33)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_SAMPLES_ADVERSARIAL = 1000  # number of adversarial samples to include in the test set (same as clean samples)
    
    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Load detectors for each attack type
    detectors = load_detectors(attack_types, device)

    # Load clean test set images and their labels
    clean_images, _ = get_test_set().get_images()
    clean_labels = np.zeros(len(clean_images), dtype=bool)
    
    for attack_type in attack_types:
        # Load adversarial images from test set:
        images_dir = "./dataset/test_set/adversarial_examples/" + attack_type
        if attack_type == "df" or attack_type == "cw":
            # For DF and CW detectors, 1000 untargeted adversarial samples are used:
            # (targeted DF is not supported, targeted CW is ineffective)
            images_dir1 = images_dir + "/untargeted/samples_plot1"
            adv_images = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL)
        else:
            # For other attacks, 1000 adversarial samples are used: 500 untargeted and 500 targeted
            images_dir1 = images_dir + "/untargeted/samples_plot1"
            images_dir2 = images_dir + "/targeted/samples_plot1"
            adv_images_untargeted = get_adversarial_images(images_dir1, NUM_SAMPLES_ADVERSARIAL//2)
            adv_images_targeted = get_adversarial_images(images_dir2, NUM_SAMPLES_ADVERSARIAL//2)
            adv_images = np.concatenate((adv_images_untargeted, adv_images_targeted), axis=0)
        
        # Create labels associated with adversarial images
        adv_labels = np.ones(len(adv_images), dtype=bool)

        # Combine clean and adversarial images and labels into final test set
        final_test_images = np.concatenate((clean_images, adv_images), axis=0)
        final_test_labels = np.concatenate((clean_labels, adv_labels), axis=0)
        
        # Call the specific detector for the attack type
        detector = detectors[attack_type]
        report, is_adversarial = detector.detect(final_test_images)

        # Calculate test metrics
        y_true = final_test_labels
        y_pred = is_adversarial
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"\nTest metrics for detector '{attack_type.upper()}':")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Create directory for saving detector plots
        plot_dir = "./plots/detectors/" + attack_type
        os.makedirs(plot_dir, exist_ok=True)

        # Create and save confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=["Clean", "Adversarial"])
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title(f"Confusion Matrix - {attack_type.upper()} Detector")
        plt.savefig(plot_dir + "/Confusion_Matrix.png")

        # Create and save ROC curve plot
        logits = np.array(report["predictions"])  # logits for "clean" and "adversarial" classes
        probs = softmax(logits, axis=1)           # convert logits to probabilities
        probs_adv = probs[:, 1]                    # probability of "adversarial" class
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probs_adv)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve - {attack_type.upper()} Detector")
        plt.legend(loc='lower right')
        plt.savefig(plot_dir + "/ROC_Curve.png")

if __name__ == "__main__":
    main()
