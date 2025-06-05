import os
import torch
import argparse
from matplotlib import pyplot as plt
from nets import setup_NN1_classifier, setup_NN2_classifier
from attacks import  FGSM, BIM, PGD, DF, CW
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from test_set import get_test_set
from utils import *

NUM_CLASSES = 8631 # number of classes in the VGGFace2 dataset

# Function that computes the maximum perturbation between original and adversarial images
def compute_max_perturbation(test_images, test_images_adv, show_distribution=False):
    # Display histogram of maximum perturbation:
    if show_distribution:
        max_pert_sample = np.max(np.abs(test_images_adv - test_images), axis=(1, 2, 3))
        plt.figure(figsize=(6, 4))
        plt.hist(max_pert_sample, bins=50, color='blue', alpha=0.7)
        plt.title('Distribution of max perturbations')
        plt.xlabel('Max perturbation')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Return the maximum perturbation:
    return np.max(np.abs(test_images_adv - test_images))


# Function that computes classifier accuracy on samples x_test (y_test contains the true classes)
def compute_accuracy(classifier, x_test, y_test):
    # Classifier prediction
    probs = classifier.predict(x_test) # contains probability for each class
    y_pred = np.argmax(probs, axis=1) # contains class with highest probability

    # Compute accuracy
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy 


# Function that computes accuracy of classifier + detectors on samples x_test (y_test contains true classes and y_adv indicates if samples are clean or adversarial)
def compute_accuracy_with_detectors(classifier, x_test, y_test, y_adv, detectors, threshold=0.5, targeted=False):
    # Compute rejected samples (0 if accepted by all detectors, 1 if rejected by at least one detector)
    rejected_samples = np.zeros(x_test.shape[0], dtype=bool) # initially all samples are accepted
    for name, detector in detectors.items(): # for each detector
        report, _ = detector.detect(x_test) # detector call
        logits = np.array(report["predictions"]) # contains logits for "clean" and "adversarial" classes
        probs = softmax(logits, axis=1) # probabilities for "clean" and "adversarial" classes
        probs_adv = probs[:, 1] # only adversarial probabilities
        is_adversarial = probs_adv > threshold # check if samples are adversarial
        detection_error = np.sum(is_adversarial != y_adv) # count detector errors
        rejected_samples = np.logical_or(is_adversarial, rejected_samples) # combine detectors with OR
        #print(f"Detector {name} made {detection_error} errors")
        
    # Compute accepted samples (1 if accepted by all detectors, 0 if rejected by at least one)
    accepted_samples = np.logical_not(rejected_samples)
    x_accepted = x_test[accepted_samples]
    #print(f"Detector {name} accepted {x_accepted.shape[0]} samples")
    y_accepted = y_test[accepted_samples]
    if isinstance(y_accepted, torch.Tensor):
        y_accepted = y_accepted.cpu().numpy()

    # Accepted samples are passed to the classifier for classification
    if x_accepted.shape[0] > 0:
        probs = classifier.predict(x_accepted) # probability for each class
        y_pred = np.argmax(probs, axis=1) # class with highest probability
        correctly_classified = np.sum(y_pred == y_accepted) # correctly classified samples
    else:
        correctly_classified = 0
        print("No samples were accepted by the detectors.")

    # Compute number of correctly detected adversarial samples (true positives)
    correctly_detected = np.sum(np.logical_and(rejected_samples, y_adv))
    
    # Compute accuracy or targeted accuracy (targeted param decides which one to compute)
    num_samples = y_test.shape[0]
    if targeted:
        targeted_accuracy = correctly_classified / num_samples # correctly_detected samples not counted because attack failed
        return targeted_accuracy
    else:    
        accuracy = (correctly_classified + correctly_detected) / num_samples # correctly_detected samples considered correct
        return accuracy


# Function that computes accuracy and max perturbation for plotting
def computing_accuracy_for_plot(classifier, clean_images, clean_labels, test_set_adversarial_dir, targeted_labels, detectors, targeted):
    acc = []
    targeted_acc = []
    detector_acc = []
    detector_targeted_acc = []

    clean_images_original = clean_images

    # Load adversarial samples
    list_imgs_adv = load_images_from_npy(test_set_adversarial_dir)        
    
    # Compute accuracy on clean data
    max_perturbations = [0.0]
    if classifier[1] == True: # if classifier requires preprocessing
        clean_images = process_images(clean_images) # preprocessing for second classifier
    acc.append(compute_accuracy(classifier[0], clean_images, clean_labels))
    if targeted:
        targeted_acc.append(compute_accuracy(classifier[0], clean_images, targeted_labels))
    if detectors is not None:
        adv_flag = np.zeros(clean_images.shape[0], dtype=bool) # samples are clean
        detector_acc.append(compute_accuracy_with_detectors(classifier[0], clean_images, clean_labels, adv_flag, detectors, targeted=targeted))        
        if targeted: 
            detector_targeted_acc.append(compute_accuracy_with_detectors(classifier[0], clean_images, targeted_labels, adv_flag, detectors, targeted=targeted)) 

    # Compute accuracy on adversarial data
    for imgs_adv in list_imgs_adv:
        max_perturbations.append(compute_max_perturbation(clean_images_original, imgs_adv))
        if classifier[1] == True:
            imgs_adv = process_images(imgs_adv) # preprocessing for second classifier   
        acc.append(compute_accuracy(classifier[0], imgs_adv, clean_labels))
        if targeted:
            targeted_acc.append(compute_accuracy(classifier[0], imgs_adv, targeted_labels))
        if detectors is not None:
            adv_flag = np.ones(len(imgs_adv), dtype=bool) # samples are adversarial
            detector_acc.append(compute_accuracy_with_detectors(classifier[0], imgs_adv, clean_labels, adv_flag, detectors, targeted=False))
            if targeted:
                detector_targeted_acc.append(compute_accuracy_with_detectors(classifier[0], imgs_adv, targeted_labels, adv_flag, detectors, targeted=True))

    return acc, targeted_acc, detector_acc, detector_targeted_acc, max_perturbations


# Function to plot the security evaluation curve: accuracy and targeted accuracy vs x (an attack-specific parameter) and max perturbation
def plot_curve(title, x_title, legend, x, max_perturbation, accuracies, security_evaluation_curve_dir, targeted=False, targeted_accuracies=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    final_legend_sx = []
    final_legend_dx = []
    base_colors = ['b', 'cyan'] # colors for classifier1 and classifier2 accuracy
    targeted_colors = ['r', 'orange'] # colors for targeted accuracy classifier1 and classifier2

    # axes[0]: Accuracy and Targeted Accuracy vs x (an attack-specific parameter)
    # axes[1]: Accuracy and Targeted Accuracy vs Max Perturbation
    for idx in range(len(accuracies)):
        # Plot accuracy:
        color = base_colors[idx]
        axes[0].plot(x, accuracies[idx], marker='o', linestyle='-', color=color)
        axes[1].plot(max_perturbation, accuracies[idx], marker='o', linestyle='-', color=color)
        final_legend_sx.append(legend[idx])
        final_legend_dx.append(legend[idx])
        # Plot targeted_accuracy:
        if targeted:
            t_color = targeted_colors[idx]
            axes[0].plot(x, targeted_accuracies[idx], marker='o', linestyle='-', color=t_color)
            axes[1].plot(max_perturbation, targeted_accuracies[idx], marker='o', linestyle='-', color=t_color)
            final_legend_sx.append(legend[idx + len(base_colors)])
            final_legend_dx.append(legend[idx + len(base_colors)])

    axes[0].set_xlabel(x_title)
    axes[0].legend(final_legend_sx, loc="upper right")
    axes[0].grid()
    axes[0].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Max Perturbation")
    axes[1].legend(final_legend_dx, loc="upper right")
    axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=1.5) # red vertical line on constraint limit
    axes[1].grid()
    axes[1].set_ylim([0.0, 1.05])
    plt.tight_layout()
    save_path = os.path.join(security_evaluation_curve_dir, title)
    os.makedirs(security_evaluation_curve_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot {title}.png saved.")
    plt.close()


# Function that generates FGSM adversarial samples (if generate_samples=True) and the related security evaluation curve.
def run_fgsm(classifier, name, test_set, detectors=None, targeted=False, target_class=None, generate_samples=False, device="cpu"):
    attack_dir = "fgsm/targeted/" if targeted else "fgsm/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir + "samples_plot1"
    security_evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir + "plot1"

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Loading clean samples
    clean_images, clean_labels = test_set.get_images()

    # Accuracy calculation varying epsilon
    plot = {
        "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
        "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations",
        "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations",
        "x_axis_name": "Epsilon"
    }

    # Generation and saving of adversarial samples (if generate_samples=True)
    if generate_samples:
        attack = FGSM(setup_NN1_classifier(device))
        i=0
        for epsilon in plot["epsilon_values"]:
            imgs_adv = attack.generate_images(clean_images, epsilon, targeted, targeted_labels)
            save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon}", test_set_adversarial_dir)
            i+=1
        print("Test adversarial examples generated and saved successfully for fgsm.")
    
    # Add the point on the x-axis corresponding to accuracy on clean data
    x_axis_value = [0.0] + plot["epsilon_values"]

    # Compute accuracy and targeted accuracy for each classifier
    accuracies = [] # list of classifiers' accuracies
    targeted_accuracies = [] # list of classifiers' targeted accuracies
    for c in classifier:
        acc, targeted_acc, detector_acc, detector_target_acc, max_perturbations = computing_accuracy_for_plot(c, clean_images, clean_labels, test_set_adversarial_dir, targeted_labels, detectors, targeted)
        accuracies.append(acc)
        targeted_accuracies.append(targeted_acc)
        if detectors is not None: 
            accuracies.append(detector_acc)
            targeted_accuracies.append(detector_target_acc)
    
    # Plot the security evaluation curves
    if detectors is None: 
        legend = ["Accuracy NN1", "Accuracy NN2", "Targeted Accuracy NN1", "Targeted Accuracy NN2"]
    else: 
        legend = ["Accuracy NN1", "Accuracy NN1 + detectors", "Targeted Accuracy NN1", "Targeted Accuracy NN1 + detectors"]
    if targeted:
        plot_curve(plot["title_targeted"], plot["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
    else:
        plot_curve(plot["title_untargeted"], plot["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)


# Function that generates BIM adversarial samples (if generate_samples=True) and the related security evaluation curve.
def run_bim(classifier, name, test_set, detectors=None, targeted=False, target_class=None, generate_samples=False, device="cpu"):
    attack_dir = "bim/targeted/" if targeted else "bim/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Loading clean samples
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Accuracy calculation varying epsilon (with fixed epsilon_step and max_iter)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0.01; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0.01; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Accuracy calculation varying epsilon_step (with fixed epsilon and max_iter)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0.1; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0.1; Max_iter=10)",
            "x_axis_name": "Epsilon Step"
        },
        # Accuracy calculation varying max_iter (with fixed epsilon and epsilon_step)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=0.1; Epsilon_step=0.01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon=0.1; Eps_step=0.01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generation and saving of adversarial samples (if generate_samples=True)
        if generate_samples:
            attack = BIM(setup_NN1_classifier(device))
            i=0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i+=1
            print("Test adversarial examples generated and saved successfully for bim.")
        
        # Add the point on the x-axis corresponding to accuracy on clean data
        if plot_name=="plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name=="plot2":
            x_axis_value = [0.0] + plot_data["epsilon_step_values"]
        elif plot_name=="plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
            
        # Compute accuracy and targeted accuracy for each classifier
        accuracies = [] # list of classifiers' accuracies
        targeted_accuracies = [] # list of classifiers' targeted accuracies
        for c in classifier:
            acc, targeted_acc, detector_acc, detector_target_acc, max_perturbations = computing_accuracy_for_plot(c, clean_images, clean_labels, test_set_adv_dir, targeted_labels, detectors, targeted)
            accuracies.append(acc)
            targeted_accuracies.append(targeted_acc)
            if detectors is not None: 
                accuracies.append(detector_acc)
                targeted_accuracies.append(detector_target_acc)

        # Plot the security evaluation curves
        if detectors is None: 
            legend = ["Accuracy NN1", "Accuracy NN2", "Targeted Accuracy NN1", "Targeted Accuracy NN2"]
        else: 
            legend = ["Accuracy NN1", "Accuracy NN1 + detectors", "Targeted Accuracy NN1", "Targeted Accuracy NN1 + detectors"]
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)
        

# Function that generates PGD adversarial samples (if generate_samples=True) and their corresponding security evaluation curve.
def run_pgd(classifier, name, test_set, detectors=None, targeted=False, target_class=None, generate_samples=False, device="cpu"):
    attack_dir = "pgd/targeted/" if targeted else "pgd/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Load clean samples
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Accuracy vs Epsilon (epsilon_step and max_iter fixed)
        "plot1": {
            "epsilon_values": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0.01; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon and Max Perturbations (Epsilon_step=0.01; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Accuracy vs Epsilon Step (epsilon and max_iter fixed)
        "plot2": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01, 0.02, 0.03, 0.04, 0.05],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0.1; Max_iter=10)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Epsilon Step and Max Perturbations (Epsilon=0.1; Max_iter=10)",
            "x_axis_name": "Epsilon Step"
        },
        # Accuracy vs Max Iter (epsilon and epsilon_step fixed)
        "plot3": {
            "epsilon_values": [0.1],
            "epsilon_step_values": [0.01],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=0.1; Epsilon_step=0.01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Epsilon=0.1; Eps_step=0.01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generate and save adversarial samples (if generate_samples=True)
        if generate_samples:
            attack = PGD(setup_NN1_classifier(device))
            i = 0
            for epsilon in plot_data["epsilon_values"]:
                for epsilon_step in plot_data["epsilon_step_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, epsilon_step, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};eps_step_{epsilon_step};max_iter_{max_iter}", test_set_adv_dir)
                        i += 1
            print("Test adversarial examples generated and saved successfully for pgd.")

        # Add baseline accuracy point for clean data on the x-axis
        if plot_name == "plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name == "plot2":
            x_axis_value = [0.0] + plot_data["epsilon_step_values"]
        elif plot_name == "plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        
        # Compute accuracy and targeted accuracy for each classifier
        accuracies = []  # list of classifiers' accuracies
        targeted_accuracies = []  # list of targeted accuracies
        for c in classifier:
            acc, targeted_acc, detector_acc, detector_target_acc, max_perturbations = computing_accuracy_for_plot(c, clean_images, clean_labels, test_set_adv_dir, targeted_labels, detectors, targeted)
            accuracies.append(acc)
            targeted_accuracies.append(targeted_acc)
            if detectors is not None: 
                accuracies.append(detector_acc)
                targeted_accuracies.append(detector_target_acc)

        # Plot security evaluation curves
        if detectors is None: 
            legend = ["Accuracy NN1", "Accuracy NN2", "Targeted Accuracy NN1", "Targeted Accuracy NN2"]
        else: 
            legend = ["Accuracy NN1", "Accuracy NN1 + detectors", "Targeted Accuracy NN1", "Targeted Accuracy NN1 + detectors"]
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)

# Function that generates DF adversarial samples (if generate_samples=True) and their corresponding security evaluation curve.
def run_df(classifier, name, test_set, detectors=None, generate_samples=False, device="cpu"):
    attack_dir = "df/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    # Load clean samples
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Accuracy vs Epsilon (nb_grads and max_iter fixed)
        "plot1": {
            "epsilon_values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            "nb_grads_values": [10],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Epsilon and Max Perturbations (Nb_grads=10; Max_iter=10)",
            "x_axis_name": "Epsilon"
        },
        # Accuracy vs Nb_grads (epsilon and max_iter fixed)
        "plot2": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [5, 10, 20, 50],
            "max_iter_values": [10],
            "title_untargeted": f"{name} - Accuracy vs Nb Grads and Max Perturbations (Epsilon=1e-2; Max_iter=10)",
            "x_axis_name": "Nb Grads"
        },
        # Accuracy vs Max_iter (epsilon and nb_grads fixed)
        "plot3": {
            "epsilon_values": [1e-2],
            "nb_grads_values": [10],
            "max_iter_values": [1, 3, 5, 7, 10],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Epsilon=1e-2; Nb_grads=10)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generate and save adversarial samples (if generate_samples=True)
        if generate_samples:
            attack = DF(setup_NN1_classifier(device))
            i = 0
            for epsilon in plot_data["epsilon_values"]:
                for nb_grads in plot_data["nb_grads_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, epsilon, nb_grads, max_iter)
                        save_images_as_npy(imgs_adv, f"{i}_eps_{epsilon};nb_grads_{nb_grads};max_iter_{max_iter}", test_set_adv_dir)
                        i += 1
            print("Test adversarial examples generated and saved successfully for df.")

        # Add baseline accuracy point for clean data on the x-axis
        if plot_name == "plot1":
            x_axis_value = [0.0] + plot_data["epsilon_values"]
        elif plot_name == "plot2":
            x_axis_value = [0] + plot_data["nb_grads_values"]
        elif plot_name == "plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]

        # Compute accuracy for each classifier
        accuracies = []  # list of classifiers' accuracies
        for c in classifier:
            acc, _, detector_acc, _, max_perturbations = computing_accuracy_for_plot(c, clean_images, clean_labels, test_set_adv_dir, None, detectors, False)
            accuracies.append(acc)
            if detectors is not None: 
                accuracies.append(detector_acc)
           
        # Plot security evaluation curves
        if detectors is None: 
            legend = ["Accuracy NN1", "Accuracy NN2", "Targeted Accuracy NN1", "Targeted Accuracy NN2"]
        else: 
            legend = ["Accuracy NN1", "Accuracy NN1 + detectors", "Targeted Accuracy NN1", "Targeted Accuracy NN1 + detectors"]
        plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)       

# Function that generates adversarial CW samples (if generate_samples=True) and the related security evaluation curve.
def run_cw(classifier, name, test_set, detectors=None, targeted=False, target_class=None, generate_samples=False, device="cpu"):
    attack_dir = "cw/targeted/" if targeted else "cw/untargeted/"
    test_set_adversarial_dir = "./dataset/test_set/adversarial_examples/" + attack_dir
    evaluation_curve_dir = "./plots/security_evaluation_curve/" + attack_dir

    if targeted:
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
    else:
        targeted_labels = None

    # Loading clean samples
    clean_images, clean_labels = test_set.get_images()

    plots = {
        # Accuracy vs confidence (fixed learning_rate and max_iter)
        "plot1": {
            "confidence_values": [0.01, 0.1, 1],
            "learning_rate_values": [0.01],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Confidence and Max Perturbations (Learning_rate=0.01; Max_iter=3)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Confidence and Max Perturbations (Learning_rate=0.01; Max_iter=3)",
            "x_axis_name": "Confidence"
        },
        # Accuracy vs learning_rate (fixed confidence and max_iter)
        "plot2": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01, 0.05, 0.1],
            "max_iter_values": [3],
            "title_untargeted": f"{name} - Accuracy vs Learning Rate and Max Perturbations (Confidence=0.1; Max_iter=3)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Learning Rate and Max Perturbations (Confidence=0.1; Max_iter=3)",
            "x_axis_name": "Learning Rate"
        },
        # Accuracy vs max_iter (fixed confidence and learning_rate)
        "plot3": {
            "confidence_values": [0.1],
            "learning_rate_values": [0.01],
            "max_iter_values": [1, 3, 5],
            "title_untargeted": f"{name} - Accuracy vs Max Iterations and Max Perturbations (Confidence=0.1; Learning_rate=0.01)",
            "title_targeted": f"{name} - Accuracy and Targeted Accuracy vs Max Iterations and Max Perturbations (Confidence=0.1; Lr=0.01)",
            "x_axis_name": "Max Iterations"
        }
    }

    for plot_name, plot_data in plots.items():
        test_set_adv_dir = test_set_adversarial_dir + "samples_" + plot_name
        security_evaluation_curve_dir = evaluation_curve_dir + plot_name
        
        # Generate and save adversarial examples (if generate_samples=True)
        if generate_samples:
            attack = CW(setup_NN1_classifier(device))
            i = 0
            for confidence in plot_data["confidence_values"]:
                for learning_rate in plot_data["learning_rate_values"]:
                    for max_iter in plot_data["max_iter_values"]:
                        imgs_adv = attack.generate_images(clean_images, confidence, learning_rate, max_iter, targeted, targeted_labels)
                        save_images_as_npy(imgs_adv, f"{i}_confidence_{confidence};learning_rate_{learning_rate};max_iter_{max_iter}", test_set_adv_dir)
                        i += 1
            print("Test adversarial examples generated and saved successfully for CW.")
        
        # Add baseline accuracy on clean data to x-axis
        if plot_name == "plot1":
            x_axis_value = [0.0] + plot_data["confidence_values"]
        elif plot_name == "plot2":
            x_axis_value = [0.0] + plot_data["learning_rate_values"]
        elif plot_name == "plot3":
            x_axis_value = [0] + plot_data["max_iter_values"]
        
        # Compute accuracy and targeted accuracy for each classifier
        accuracies = []
        targeted_accuracies = []
        for c in classifier:
            acc, targeted_acc, detector_acc, detector_target_acc, max_perturbations = computing_accuracy_for_plot(c, clean_images, clean_labels, test_set_adv_dir, targeted_labels, detectors, targeted)
            accuracies.append(acc)
            targeted_accuracies.append(targeted_acc)
            if detectors is not None:
                accuracies.append(detector_acc)
                targeted_accuracies.append(detector_target_acc)

        # Plot security evaluation curves
        if detectors is None:
            legend = ["Accuracy NN1", "Accuracy NN2", "Targeted Accuracy NN1", "Targeted Accuracy NN2"]
        else:
            legend = ["Accuracy NN1", "Accuracy NN1 + detectors", "Targeted Accuracy NN1", "Targeted Accuracy NN1 + detectors"]
        if targeted:
            plot_curve(plot_data["title_targeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir, targeted, targeted_accuracies)
        else:
            plot_curve(plot_data["title_untargeted"], plot_data["x_axis_name"], legend, x_axis_value, max_perturbations, accuracies, security_evaluation_curve_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_name", type=str, default="NN1", choices=["NN1", "NN2", "NN1 + detectors"], help="Classifier to test")
    parser.add_argument('--generate_samples', type=bool, default=False, help='True to generate adversarial images and evaluation curves, False to only generate the evaluation curves')
    args = parser.parse_args()
    
    classifier_name = args.classifier_name
    generate_samples = args.generate_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    attack_types = ["fgsm", "bim", "pgd", "df", "cw"]

    # Load clean test set samples and their labels
    test_set = get_test_set()
    clean_images, clean_labels = test_set.get_images()

    # Setup list of classifiers to evaluate: pair (classifier, preprocess_flag)
    classifiers = [[setup_NN1_classifier(device), False]]
    detectors = None
    if classifier_name == "NN2":
        classifiers.append([setup_NN2_classifier(device), True])
    elif classifier_name == "NN1 + detectors":
        detectors = load_detectors(attack_types, device)

    # Evaluation on clean samples
    if classifier_name == "NN1":
        accuracy_clean = compute_accuracy(classifiers[0][0], clean_images, clean_labels)
        print(f"Accuracy of classifier {classifier_name} on clean data: {accuracy_clean:.3f}")
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
        targeted_accuracy_clean = compute_accuracy(classifiers[0][0], clean_images, targeted_labels)
        print(f"Targeted accuracy of classifier {classifier_name} on clean data: {targeted_accuracy_clean:.3f}")
    elif classifier_name == "NN2":
        clean_images = process_images(clean_images)
        accuracy_clean = compute_accuracy(classifiers[1][0], clean_images, clean_labels)
        print(f"Accuracy of classifier {classifier_name} on clean data: {accuracy_clean:.3f}")
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(clean_labels.size, dtype=torch.long)
        targeted_accuracy_clean = compute_accuracy(classifiers[1][0], clean_images, targeted_labels)
        print(f"Targeted accuracy of classifier {classifier_name} on clean data: {targeted_accuracy_clean:.3f}")
    elif classifier_name == "NN1 + detectors":
        adv_labels = np.zeros(clean_images.shape[0], dtype=bool)
        accuracy_clean = compute_accuracy_with_detectors(classifiers[0][0], clean_images, clean_labels, adv_labels, detectors, targeted=False)
        print(f"Accuracy of classifier {classifier_name} on clean data: {accuracy_clean:.3f}")
        target_class_label = "Cristiano_Ronaldo"
        target_class = test_set.get_true_label(target_class_label)
        targeted_labels = target_class * torch.ones(len(test_set), dtype=torch.long)
        targeted_accuracy_clean = compute_accuracy_with_detectors(classifiers[0][0], clean_images, targeted_labels, adv_labels, detectors, targeted=True)
        print(f"Targeted accuracy of classifier {classifier_name} on clean data: {targeted_accuracy_clean:.3f}")
    else:
        print(f"Invalid classifier {classifier_name} (use 'NN1', 'NN2' or 'NN1 + detectors').")    
        return

    # Evaluation on adversarial samples (security evaluation curve)

    # Untargeted attacks:
    if "fgsm" in attack_types:
        run_fgsm(classifiers, classifier_name, test_set, detectors, generate_samples=generate_samples, device=device)
    if "bim" in attack_types:
        run_bim(classifiers, classifier_name, test_set, detectors, generate_samples=generate_samples, device=device)
    if "pgd" in attack_types:
        run_pgd(classifiers, classifier_name, test_set, detectors, generate_samples=generate_samples, device=device)
    if "df" in attack_types:
        run_df(classifiers, classifier_name, test_set, detectors, generate_samples=generate_samples, device=device)
    if "cw" in attack_types:
        run_cw(classifiers, classifier_name, test_set, detectors, generate_samples=generate_samples, device=device)
    # Targeted attacks:
    if "fgsm" in attack_types:
        run_fgsm(classifiers, classifier_name, test_set, detectors, True, target_class, generate_samples, device)
    if "bim" in attack_types:
        run_bim(classifiers, classifier_name, test_set, detectors, True, target_class, generate_samples, device)
    if "pgd" in attack_types:
        run_pgd(classifiers, classifier_name, test_set, detectors, True, target_class, generate_samples, device)
    if "cw" in attack_types:
        run_cw(classifiers, classifier_name, test_set, detectors, True, target_class, generate_samples, device)

if __name__ == "__main__":
    main()