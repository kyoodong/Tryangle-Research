import torch.nn as nn
import torch

from background_classification.data.config import cfg
from background_classification.model.backbone.efficientnet import EfficientNet


class BGClassification(nn.Module):

    def __init__(self):
        super().__init__()

        # include_top이 False이면 EfficientNet을 실질적으로 실행시킬 때 fc 부분을 실행을 안 함
        self._backbone = EfficientNet.from_pretrained(cfg.backbone.model_name,
                                                     num_classes=cfg.num_classes,
                                                     include_top=cfg.backbone.include_top)

        out_channels = self._backbone._conv_head.out_channels
        self._dropout = nn.Dropout(cfg.dropout_rate)
        self._fc = nn.Linear(out_channels, cfg.num_classes)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        self.load_state_dict(state_dict)

    def forward(self, x):

        x = self._backbone(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x
