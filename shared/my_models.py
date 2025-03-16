import torch.nn as nn
import torchvision.models as models
import copy
import torch
import torch.nn.functional as F

def get_tabular_model(X_train, dropout=0.9):
    tabular_model = nn.Sequential(
        nn.Linear(X_train.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    return tabular_model


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(1024, 1)
    
    def forward(self, x):
        return self.model(x)


class ConcatModel(nn.Module):
    def __init__(self, img_model, tabular_model):
        super(ConcatModel, self).__init__()
        # Make copies in case the modules get trained later.
        img_model = copy.deepcopy(img_model)
        tabular_model = copy.deepcopy(tabular_model)
        self.img_module = nn.Sequential(*list(img_model.model.children())[:-1])
        for param in self.img_module.parameters():
            param.requires_grad = False

        self.tab_module = tabular_model[:-1]
        for param in self.tab_module.parameters():
            param.requires_grad = False

        # Add the new classification layers.
        self.fc1 = nn.Linear(128+1024, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,1)
    
    def forward(self, img, X_tab):
        img_out = self.img_module(img)
        img_out = F.adaptive_avg_pool2d(img_out, (1, 1)).view(img_out.size(0), -1)
        tab_out = self.tab_module(X_tab)
        combined = torch.cat([img_out, tab_out], dim=1)
        combined = self.fc1(combined)
        combined = self.bn1(combined)
        combined = self.dropout(combined)
        combined = nn.ReLU()(combined)
        return self.fc2(combined)
    
class CrossAttentionModel(nn.Module):
    def __init__(self, img_model, tabular_model):
        super(CrossAttentionModel, self).__init__()
        # Make copies in case the modules get trained later.
        img_model = copy.deepcopy(img_model)
        tabular_model = copy.deepcopy(tabular_model)
        self.img_module = nn.Sequential(*list(img_model.model.children())[:-1])
        for param in self.img_module.parameters():
            param.requires_grad = False

        self.tab_module = tabular_model[:-1]
        for param in self.tab_module.parameters():
            param.requires_grad = False

        # Add the new classification layers.
        self.tab_to_img_dim = nn.Linear(128, 1024)
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(1024, 1)
    
    def forward(self, img, X_tab):
        img_out = self.img_module(img)
        img_out = F.adaptive_avg_pool2d(img_out, (1, 1)).view(img_out.size(0), -1)
        tab_out = self.tab_module(X_tab)
        tab_out = self.tab_to_img_dim(tab_out)
        tab_out = tab_out.unsqueeze(1)
        img_out = img_out.unsqueeze(1)
        x, _ = self.attention(tab_out, img_out, img_out)
        x = x.squeeze(1)
        return self.fc(x)
    

class SelfAttentionModel(nn.Module):
    def __init__(self, img_model, tabular_model):
        super(SelfAttentionModel, self).__init__()
        # Make copies in case the modules get trained later.
        img_model = copy.deepcopy(img_model)
        tabular_model = copy.deepcopy(tabular_model)
        self.img_module = nn.Sequential(*list(img_model.model.children())[:-1])
        for param in self.img_module.parameters():
            param.requires_grad = False

        self.tab_module = tabular_model[:-1]
        for param in self.tab_module.parameters():
            param.requires_grad = False

        # Add the new classification layers.
        # self.tab_to_img_dim = nn.Linear(128, 1024)
        self.attention = nn.MultiheadAttention(embed_dim=1024+128, num_heads=8, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(1024+128, 1)
    
    def forward(self, img, X_tab):
        img_out = self.img_module(img)
        img_out = F.adaptive_avg_pool2d(img_out, (1, 1)).view(img_out.size(0), -1)
        tab_out = self.tab_module(X_tab)
        # tab_out = self.tab_to_img_dim(tab_out)
        # tab_out = tab_out.unsqueeze(1)
        # img_out = img_out.unsqueeze(1)
        combined = torch.cat([img_out, tab_out], dim=1)
        x, _ = self.attention(combined, combined, combined)
        x = x.squeeze(1)
        return self.fc(x)