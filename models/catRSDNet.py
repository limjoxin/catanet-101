import torch
from torch import nn
from typing import TypeVar
from torchvision.models import densenet169, DenseNet169_Weights
import numpy as np

T = TypeVar('T', bound='Module')

class MtFc(nn.Module):
    """Multi-task Fully Connected layers for step and RSD prediction"""
    def __init__(self, n_in, n_step, n_experience, n_rsd=1):
        super(MtFc, self).__init__()
        self.fc1 = nn.Linear(n_in, n_step)    # Step prediction
        # self.fc2 = nn.Linear(n_in, n_experience)
        self.fc3 = nn.Linear(n_in, n_rsd)     # RSD prediction

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
    """Modified to only handle step prediction"""
    def __init__(self, n_in, n_1, n_2):
        super(Parallel_fc, self).__init__()
        self.fc1 = nn.Linear(n_in, n_1)  # Step prediction
        # self.fc2 = nn.Linear(n_in, n_2)

    def forward(self, X):
        return self.fc1(X.clone())

class CatRSDNet(nn.Module):
    """
    Modified CatRSDNet focusing on step prediction and RSD calculation

    Parameters:
    -----------
    n_step_classes: int, optional
      number of surgical step classes, default 11
    n_expertise_classes: int, optional
      kept for compatibility but not used
    max_len: int, optional
      maximum length of videos to process in minutes, default 20 min
    """
    def __init__(self, n_step_classes=13, n_expertise_classes=2, max_len=20):
        super(CatRSDNet, self).__init__()
        feature_size = 1664
        self.max_len = max_len
        # Initialize CNN - keeping n_expertise_classes for compatibility
        self.cnn = self.initCNN(feature_size, n_classes_1=n_step_classes, n_classes_2=n_expertise_classes)
        
        self.rnn = Rnn_Model(input_size=feature_size, output_size=n_step_classes)
        self.rnn.fc = MtFc(128, n_step_classes, n_expertise_classes)

    def train(self: T, mode: bool = True) -> T:
        super(CatRSDNet, self).train(mode)

    def freeze_cnn(self, freeze=True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_rnn(self, freeze=True):
        for param in self.rnn.parameters():
            param.requires_grad = not freeze

    def initCNN(self, feature_size, n_classes_1, n_classes_2):
        """Initialize CNN with modified classifier for step prediction only"""
        cnn = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        tmp_conv_weights = cnn.features.conv0.weight.data.clone()
        cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
        cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights

        # Modified classifier for step prediction only
        cnn.classifier = Parallel_fc(n_in=feature_size, n_1=n_classes_1, n_2=n_classes_2)
        return cnn

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
        """
        Modified forward pass for step prediction and RSD only

        Parameters
        ----------
        X: torch.tensor, shape(batch_size, k, 224, 224)
            input image tensor, k=3 if elapsed_time provided otherwise k=4 with time_channel
        elapsed_time: float, optional
            elapsed time from start of surgery in minutes
        stateful: bool, optional
            true -> store last RNN state, default false

        Returns
        -------
        tuple:
            - step_prediction: torch.tensor, shape(batch_size, n_step_classes)
            - rsd_predictions: torch.tensor, shape(batch_size,)
        """
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