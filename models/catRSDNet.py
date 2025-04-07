import torch
from torch import nn
from typing import TypeVar
from torch.nn import Module
from torchvision.models import densenet169, DenseNet169_Weights, resnet50, ResNet50_Weights
import numpy as np

T = TypeVar('T', bound='Module')

class MtFc(nn.Module):
    def __init__(self, n_in, n_step, n_rsd=1):
        super(MtFc, self).__init__()
        self.fc1 = nn.Linear(n_in, n_step)
        self.fc3 = nn.Linear(n_in, n_rsd)

    def forward(self, x):
        step = self.fc1(x.clone())
        rsd = self.fc3(x.clone())
        return step, rsd

class Rnn_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Rnn_Model, self).__init__()
        self.rnn_cell = nn.LSTM(input_size=input_size,
                           hidden_size=128,
                           num_layers=2,
                           dropout=0.0,
                           batch_first=True)
        if isinstance(self.rnn_cell.hidden_size, list):
            hidden_size = self.rnn_cell.hidden_size[-1]
        else:
            hidden_size = self.rnn_cell.hidden_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.last_state = None

    def forward(self, X, stateful=False, ret_feature=False):
        if stateful:
            init_state = self.last_state
        else:
            init_state = None

        y, last_state = self.rnn_cell(X, init_state)
        self.last_state = (last_state[0].detach(), last_state[1].detach())
        y = y.squeeze(0)
        y_out = self.fc(y)
        if ret_feature:
            return y_out, y
        return y_out

class Parallel_fc(nn.Module):
    def __init__(self, n_in, n_1):
        super(Parallel_fc, self).__init__()
        self.fc1 = nn.Linear(n_in, n_1)

    def forward(self, X):
        return self.fc1(X.clone())


class EnhancedCatRSDNet(nn.Module):
    """
    Enhanced CNN model based on a pre-trained ResNet50 backbone.
    This model is designed for improved performance on cataract surgery tool detection.
    """
    def __init__(self, n_classes=13, dropout_rate=0.5):
        super(EnhancedCatRSDNet, self).__init__()
        
        # Load the pretrained model
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv layer to accept 4 channels (RGB + timestamp)
        original_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            4,  # Change from 3 to 4 channels
            original_layer.out_channels, 
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=(original_layer.bias is not None)
        )
        
        # Initialize the 4th channel with mean of RGB channels
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3, :, :] = original_layer.weight
            self.backbone.conv1.weight[:, 3:4, :, :] = torch.mean(original_layer.weight, dim=1, keepdim=True)
        
        # Create features extractor (all layers except classifier)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),  # ResNet50 outputs 2048-dimensional features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        # Extract features using the modified backbone
        features = self.features(x)
        # Classify using the custom head
        output = self.classifier(features)
        return output
    
    def set_as_feature_extractor(self):
        """
        Convert the model to a feature extractor by removing the classifier.
        Used when features are needed for RNN input.
        """
        self.classifier = nn.Identity()
        
    def freeze_early_layers(self, freeze=True):
        """
        Freeze early layers of the ResNet backbone while keeping later layers trainable.
        
        Args:
            freeze (bool): If True, freeze the early layers; if False, make them trainable.
        """
        # Freeze or unfreeze conv1, bn1, layer1, layer2
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = not freeze
                
        print(f"Early backbone layers frozen: {freeze}")
        
    def freeze_late_layers(self, freeze=True):
        """
        Freeze late (deeper) layers of the ResNet backbone while keeping early layers trainable.
        
        Args:
            freeze (bool): If True, freeze the late layers; if False, make them trainable.
        """
        # Freeze or unfreeze layer3, layer4
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ['layer3', 'layer4']):
                param.requires_grad = not freeze
                
        # Also handle the classifier
        for param in self.classifier.parameters():
            param.requires_grad = not freeze
                
        print(f"Late backbone layers frozen: {freeze}")


class CatRSDNet(nn.Module):
    def __init__(self, n_step_classes=13, max_len=20):
        super(CatRSDNet, self).__init__()
        feature_size = 1664
        self.max_len = max_len
        self.cnn = self.initCNN(feature_size, n_classes_1=n_step_classes)
        
        self.rnn = Rnn_Model(input_size=feature_size, output_size=n_step_classes)
        self.rnn.fc = MtFc(128, n_step_classes, n_rsd=1)

    def train(self: T, mode: bool = True) -> T:
        super(CatRSDNet, self).train(mode)

    def freeze_cnn(self, freeze=True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze=True):
        for param in self.rnn.parameters():
            param.requires_grad = not freeze
    
    def freeze_early_cnn_layers(self, freeze=True):
        """
        Freeze early layers of the CNN backbone while keeping later layers trainable.
        This is useful for fine-tuning only the deeper layers which contain more task-specific features.
        
        Args:
            freeze (bool): If True, freeze the early layers; if False, make them trainable.
        """
        # Assuming the CNN is a ResNet or similar architecture with named modules
        # We'll freeze all layers up to the final block
        
        # First, check what type of CNN we have to determine the freezing strategy
        cnn_name = self.cnn.__class__.__name__.lower()
        
        if hasattr(self.cnn, 'layer1') and hasattr(self.cnn, 'layer2') and hasattr(self.cnn, 'layer3') and hasattr(self.cnn, 'layer4'):
            # ResNet-like structure with named layer blocks
            print(f"Identified ResNet-like architecture with named layers")
            
            # Freeze or unfreeze early layers (conv1, bn1, layer1, layer2)
            for name, param in self.cnn.named_parameters():
                if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = not freeze
                    
            print(f"Early CNN layers frozen: {freeze}")
            
        elif hasattr(self.cnn, 'features') and hasattr(self.cnn, 'classifier'):
            # VGG or similar structure with features and classifier
            print(f"Identified VGG-like architecture with features and classifier")
            
            # For VGG-like networks, freeze first 2/3 of the feature extractor
            features = list(self.cnn.features)
            freeze_until = len(features) * 2 // 3
            
            for i, param in enumerate(self.cnn.features.parameters()):
                if i < freeze_until:
                    param.requires_grad = not freeze
                    
            print(f"Early CNN layers frozen: {freeze} (first {freeze_until}/{len(features)} feature layers)")
            
        else:
            # Generic approach - freeze first half of all parameters
            print("Using generic approach to freeze early layers")
            
            all_params = list(self.cnn.named_parameters())
            freeze_until = len(all_params) // 2
            
            for i, (name, param) in enumerate(all_params):
                if i < freeze_until:
                    param.requires_grad = not freeze
                    
            print(f"Early CNN layers frozen: {freeze} (first {freeze_until}/{len(all_params)} parameters)")
        
    def freeze_late_cnn_layers(self, freeze=True):
        """
        Freeze late (deeper) layers of the CNN backbone while keeping early layers trainable.
        This is the opposite of freeze_early_cnn_layers.
        
        Args:
            freeze (bool): If True, freeze the late layers; if False, make them trainable.
        """
        # Similar approach as freeze_early_cnn_layers but for later layers
        
        cnn_name = self.cnn.__class__.__name__.lower()
        
        if hasattr(self.cnn, 'layer1') and hasattr(self.cnn, 'layer2') and hasattr(self.cnn, 'layer3') and hasattr(self.cnn, 'layer4'):
            # ResNet-like structure
            for name, param in self.cnn.named_parameters():
                if any(x in name for x in ['layer3', 'layer4', 'fc']):
                    param.requires_grad = not freeze
                    
            print(f"Late CNN layers frozen: {freeze}")
            
        elif hasattr(self.cnn, 'features') and hasattr(self.cnn, 'classifier'):
            # VGG or similar structure
            features = list(self.cnn.features)
            freeze_from = len(features) * 2 // 3
            
            for i, param in enumerate(self.cnn.features.parameters()):
                if i >= freeze_from:
                    param.requires_grad = not freeze
                    
            # Also handle the classifier
            for param in self.cnn.classifier.parameters():
                param.requires_grad = not freeze
                    
            print(f"Late CNN layers frozen: {freeze} (last {len(features) - freeze_from}/{len(features)} feature layers and classifier)")
            
        else:
            # Generic approach
            all_params = list(self.cnn.named_parameters())
            freeze_from = len(all_params) // 2
            
            for i, (name, param) in enumerate(all_params):
                if i >= freeze_from:
                    param.requires_grad = not freeze
                    
            print(f"Late CNN layers frozen: {freeze} (last {len(all_params) - freeze_from}/{len(all_params)} parameters)")

    @staticmethod
    def get_trainable_cnn_params(model):
        """
        Get all trainable parameters from the CNN.
        This is useful for creating parameter groups with different learning rates.
        
        Args:
            model: The CatRSDNet model
            
        Returns:
            List of trainable parameters from the CNN
        """
        return [param for param in model.cnn.parameters() if param.requires_grad]

    def initCNN(self, feature_size, n_classes_1):
        """Initialize CNN with modified classifier for step prediction only"""
        # Load model with pre-trained weights
        cnn = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        
        # Get the current state dictionary
        state_dict = cnn.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            # Check if the key follows the pattern with dots (norm.1, conv.1)
            if '.norm.' in key or '.conv.' in key:
                new_key = key.replace('norm.1', 'norm1').replace('conv.1', 'conv1').replace('norm.2', 'norm2').replace('conv.2', 'conv2')
                new_state_dict[new_key] = value
            else:
                # Keep keys that don't need transformation
                new_state_dict[key] = value
        
        cnn_fixed = densenet169(weights=None)
        cnn_fixed.load_state_dict(new_state_dict)
        
        tmp_conv_weights = cnn_fixed.features.conv0.weight.data.clone()
        cnn_fixed.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn_fixed.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
        cnn_fixed.features.conv0.weight.data[:, 3, :, :] = mean_weights
        
        # Modified classifier for step prediction only
        cnn_fixed.classifier = Parallel_fc(n_in=feature_size, n_1=n_classes_1)
        return cnn_fixed

    def set_cnn_as_feature_extractor(self):
        self.cnn.classifier = torch.nn.Identity()

    def forwardCNN(self, images, elapsed_time):
        # Convert elapsed time in minutes to value between 0 and 1
        rel_time = elapsed_time/self.max_len
        rel_time = (torch.ones_like(images[:, 0]).unsqueeze(1) * rel_time[:, np.newaxis, np.newaxis, np.newaxis]).to(images.device)
        images_and_elapsed_time = torch.cat((images, rel_time), 1).float()
        return self.cnn(images_and_elapsed_time)

    def forwardRNN(self, X, stateful=False, skip_features=False):
        """Modified to return only step prediction and RSD"""
        if skip_features:
            features = X
        else:
            self.set_cnn_as_feature_extractor()
            features = self.cnn(X)
            features = features.unsqueeze(0)
        step_prediction, rsd_predictions = self.rnn(features, stateful=stateful)
        return step_prediction, rsd_predictions

    def forward(self, X, elapsed_time=None, stateful=False):
        self.set_cnn_as_feature_extractor()
        if elapsed_time is not None:
            assert X.shape[1] == 3, 'provide images with RGB channels only if elapsed time is set.'
            features = self.forwardCNN(X, elapsed_time.view(-1))
        else:
            assert X.shape[1] == 4, 'provide images with RGB+Timestamp if elapsed time is not set.'
            features = self.cnn(X)
        features = features.unsqueeze(0)
        step_prediction, rsd_predictions = self.rnn(features, stateful=stateful)
        return step_prediction, rsd_predictions