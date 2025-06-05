import pickle
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.optim import Adam
from art.estimators.classification import PyTorchClassifier
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1 # rete NN1

#################################################################################################################################################
# Codice per la rete NN2
# https://github.com/cydonia999/VGGFace2-pytorch/blob/master/models/senet.py
# https://github.com/cydonia999/VGGFace2-pytorch/blob/master/utils.py

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # SENet
        compress_rate = 16
        
        self.conv4 = nn.Conv2d(planes * 4, planes * 4 // compress_rate, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(planes * 4 // compress_rate, planes * 4, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ## senet
        out2 = F.avg_pool2d(out, kernel_size=out.size(2))
        out2 = self.conv4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.sigmoid(out2)
        

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out2 * out + residual
        out = self.relu(out)
        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def senet50(**kwargs):
    """Constructs a SENet-50 model.
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model



def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

#################################################################################################################################################

NUM_CLASSES = 8631  # numero di classi nel dataset VGGFace2

### RETE NN1 ###

# Funzione per caricare il modello InceptionResnetV1 (NN1)
def get_NN1(device="cpu"):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    model.to(device)
    model.classify = True
    print("Modello NN1 caricato correttamente.")
    return model

# Setup del classificatore NN1
def setup_NN1_classifier(device):
    NN1 = get_NN1(device) # modello
    NN1_classifier = PyTorchClassifier(
        model=NN1,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(NN1.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        clip_values=(-1.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return NN1_classifier


### RETE NN2 ###

# Funzione per caricare il modello SENet (NN2)
def get_NN2(device="cpu", model_path='./models/senet50_ft_weight.pkl'):
    model = senet50(num_classes=NUM_CLASSES, include_top=True)
    load_state_dict(model, model_path)
    model.to(device)
    model.eval()
    print("Modello NN2 caricato correttamente.")
    return model

# Setup del classificatore NN2
def setup_NN2_classifier(device):
    NN2 = get_NN2(device) # modello
    NN2_classifier = PyTorchClassifier(
        model=NN2,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=Adam(NN2.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=NUM_CLASSES,
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return NN2_classifier


### RETE DETECTORS ###

# Rete dei detectors usati per classificare le immagini come clean o adversarial
class AdversarialDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # output: [clean, adversarial]
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)
    
# Funzione che crea e restituisce un detector
def get_detector(device="cpu"):
    backbone = InceptionResnetV1(classify=False) # i detectors utilizzano InceptionResnetV1 come backbone
    for param in backbone.parameters():
        param.requires_grad = True # rende tutti i parametri della backbone addestrabili
    detector = AdversarialDetector(backbone)
    detector.to(device)
    return detector

# Setup del classificatore del detector
def setup_detector_classifier(device):
    detector = get_detector(device)
    classifier = PyTorchClassifier(
        model=detector,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(detector.parameters(), lr=1e-4),
        input_shape=(3, 224, 224),
        channels_first=True,
        nb_classes=2,
        clip_values=(-1.0, 1.0), # accetta valori compresi tra -1.0 e 1.0
        device_type="gpu" if torch.cuda.is_available() else "cpu"
    )
    return classifier