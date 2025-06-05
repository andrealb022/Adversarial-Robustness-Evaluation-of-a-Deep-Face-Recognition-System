# Adversarial-Robustness-Evaluation-of-a-Deep-Face-Recognition-System
Evaluated the robustness of a deep face recognition model (InceptionResNetV1) against adversarial attacks. Tested multiple attack types, analyzed transferability, and implemented a defense system using specialized detectors to improve security while preserving accuracy. Based on dataset VGGFACE2.

Project Structure
-----------------
.
├── attacks/                  # Contains adversarial attack implementations using ART (FGSM, BIM, PGD, DeepFool, CarliniWagner)
├── detectors_test.py         # Script to test adversarial detectors on each of the five attacks
├── detectors_train_set.py    # Script to generate a training set for the detectors using the VGGFace2 dataset
├── detectors_train.py        # Script to train detectors for each attack. Outputs 5 models saved in the `models/` directory
├── nets.py                   # Defines all neural networks used:
│   ├── NN1: InceptionResnetV1 from FaceNet
│   └── NN2: SEnet50 pretrained on VGGFace2
├── security_evaluation_curve.py  # Generates adversarial test sets and security evaluation curves for each attack.
│                               # The resulting plots are saved under `plots/security_evaluation_curve/`
├── test_set.py               # Randomly samples a smaller subset from VGGFace2 for faster testing and analysis
├── utils.py                  # Utility functions for loading images, saving data, plotting curves, etc.
├── dataset/
│   ├── vggface2_train/train/ # ⚠️ You must manually download and place the VGGFace2 dataset here
│   └── test_set/
│       ├── clean/
│       └── adversarial_examples/
├── models/                   # Trained detectors and NN2 weights go here (see below)
└── plots/                    # Stores accuracy and detector performance plots


Dependencies
------------

Ensure you have the following Python packages installed:

    pip install torch torchvision scikit-learn matplotlib adversarial-robustness-toolbox facenet-pytorch

Neural Networks
---------------

- NN1: InceptionResnetV1 from the `facenet_pytorch` package (used for main classification tasks).
- NN2: SEnet50 trained on VGGFace2 (used to test attack transferability).
- Detectors: Trained binary classifiers to detect adversarial examples for each attack.

Pretrained Weights
------------------

Download the NN2 network weights file (`senet50_ft_weight.pkl`) from the original repo:

VGGFace2 PyTorch (Official):  
https://github.com/cydonia999/VGGFace2-pytorch

Place the weight file into the `models/` directory.

Dataset Setup
-------------

Download and extract the VGGFace2 dataset:

- Place the training set at:  
  ./dataset/vggface2_train/train

Note: This project assumes the dataset is preprocessed as required by the models.

Running the Pipeline
--------------------

1. Create a test set:  
       python test_set.py

2. Generate adversarial train sets for detectors:  
       python detectors_train_set.py

3. Train the detectors:  
       python detectors_train.py

4. Evaluate detectors:  
       python detectors_test.py

5. Generate adversarial test samples and evaluation curves:  
       python security_evaluation_curve.py

Outputs
-------

- Adversarial samples:  
  Stored under dataset/test_set/adversarial_examples/{attack}/.

- Trained models:  
  Saved in models/.

- Evaluation curves:  
  Plots showing accuracy and perturbation performance are saved in plots/security_evaluation_curve/.

Notes
-----

- The pipeline supports 5 attacks: FGSM, BIM, PGD, DeepFool, CarliniWagner.
- Security evaluation curves plot classification accuracy against parameters like epsilon, step size, and iterations.
- Detectors are trained and evaluated for each individual attack to measure their detection efficacy.
