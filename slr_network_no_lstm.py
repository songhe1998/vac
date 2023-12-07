import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from modules import TemporalConv

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x):
        attention_weights = torch.softmax(self.fc(x), dim=-1)
        print(attention_weights)
        feats = x * attention_weights
        return feats


class Iso(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(Iso, self).__init__()
        self.conv2d = resnet18(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   num_classes=num_classes)
        self.loss_fn = torch.nn.CTCLoss(reduction='none', zero_infinity=False)


    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x):

        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise = self.masked_bn(inputs, len_x)
        framewise = framewise.reshape(batch, temp, -1)
        #framewise = self.attn(framewise)
        framewise = framewise.transpose(1, 2)


        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        logits = conv1d_outputs['conv_logits'].permute(1,0,2)
        #logits = torch.mean(logits, dim=1)

        return {
            "feat_len": lgt,
            "conv_logits": logits,
            "conv_sents": conv_pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = self.loss_fn(ret_dict["conv_logits"].permute(1,0,2).log_softmax(-1),
                                              label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                              label_lgt.cpu().int()).mean()

if __name__ == '__main__':
    model = Iso(10,512)
    inputs = torch.randn(2,44,3,256,256)
    output = model(inputs, torch.tensor([44,32]))
    print(output.shape)

