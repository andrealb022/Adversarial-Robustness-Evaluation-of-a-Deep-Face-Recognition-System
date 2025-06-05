import numpy as np
import torch
from abc import ABC, abstractmethod
from utils import *
from tqdm import tqdm
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, DeepFool, CarliniLInfMethod

NUM_CLASSES = 8631  # number of classes in the VGGFace2 dataset

class AdversarialAttack(ABC):
    def __init__(self, classifierNN1):
        self.classifierNN1 = classifierNN1  # attacks are performed on the NN1 classifier

    @abstractmethod
    def generate_images(self, images, targeted=False, target_class=0):
        pass

# Class to manage FGSM (Fast Gradient Sign Method) attack
class FGSM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, targeted=False, targeted_labels=None):
        attack = FastGradientMethod(estimator=self.classifierNN1, eps=epsilon, targeted=targeted)
        if targeted:
            # Targeted attack:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # Untargeted attack:
            return attack.generate(images)

# Class to manage BIM (Basic Iterative Method) attack
class BIM(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = BasicIterativeMethod(estimator=self.classifierNN1, eps=epsilon, eps_step=epsilon_step, max_iter=max_iter, targeted=targeted)
        if targeted:
            # Targeted attack:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # Untargeted attack:
            return attack.generate(images)

# Class to manage PGD (Projected Gradient Descent) attack
class PGD(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, epsilon_step, max_iter, targeted=False, targeted_labels=None):
        attack = ProjectedGradientDescent(
            estimator=self.classifierNN1,
            eps=epsilon,
            eps_step=epsilon_step,
            max_iter=max_iter,
            num_random_init=5,
            targeted=targeted
        )
        if targeted:
            # Targeted attack:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # Untargeted attack:
            return attack.generate(images)

# Class to manage DF (DeepFool) attack
class DF(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, epsilon, nb_grads, max_iter):
        # Untargeted attack (ART library does not support targeted DeepFool):
        attack = DeepFool(classifier=self.classifierNN1, epsilon=epsilon, nb_grads=nb_grads, max_iter=max_iter, verbose=False)
        # For efficiency, images are processed one at a time:
        adv_images = []
        batch_size = 1
        for i in tqdm(range(0, len(images), batch_size), desc="DeepFool"):
            batch = images[i:i+batch_size]
            adv_images.append(attack.generate(batch))
        adv_images = np.concatenate(adv_images, axis=0)
        return adv_images

# Class to manage CW (Carlini-Wagner) attack
class CW(AdversarialAttack):
    def __init__(self, classifierNN1):
        super().__init__(classifierNN1)

    def generate_images(self, images, confidence, learning_rate, max_iter, targeted=False, targeted_labels=None):
        attack = CarliniLInfMethod(
            classifier=self.classifierNN1,
            confidence=confidence,
            learning_rate=learning_rate,
            max_iter=max_iter,
            initial_const=0.1,
            targeted=targeted
        )
        if targeted:
            # Targeted attack:
            one_hot_targeted_labels = torch.nn.functional.one_hot(targeted_labels, NUM_CLASSES).numpy()
            return attack.generate(images, one_hot_targeted_labels)
        else:
            # Untargeted attack:
            return attack.generate(images)