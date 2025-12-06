import torch
import torch.nn as nn
import torch.nn.functional as F

#Keep original DeepDTA backbone for ligand and protein branches
class DeepDTAConv1d(nn.Module):
    def __init__(self, vocab_size, channel, kernel_size, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, channel, kernel_size)
        self.bn1 = nn.BatchNorm1d(channel)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size)
        self.bn2 = nn.BatchNorm1d(channel*2)
        self.conv3 = nn.Conv1d(channel*2, channel*3, kernel_size)
        self.bn3 = nn.BatchNorm1d(channel*3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output_dim = channel * 3

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x.squeeze(-1)

#Lightweight conv1d for stereo branch
class SimpleConv1d(nn.Module):
    def __init__(self, vocab_size, channel, kernel_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, channel, kernel_size)
        self.bn = nn.BatchNorm1d(channel)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x.squeeze(-1)


#doesn't help performance
# class AttentionFusion(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.attn = nn.Parameter(torch.ones(len(dims)) * 0.1)

#     def forward(self, feats):
#         w = F.softmax(self.attn, dim=0)
#         return sum(w_i * f_i for w_i, f_i in zip(w, feats))

#Full StereoDTA model
class StereoDTA(nn.Module):
    def __init__(self, pro_vocab_size, lig_vocab_size, stereo_vocab_size,
                 channel=64, protein_kernel=8, ligand_kernel=8,
                 descriptor_size=200):

        super().__init__()

        #DeepDTA backbone
        self.protein_conv = DeepDTAConv1d(pro_vocab_size, channel, protein_kernel)
        self.ligand_conv  = DeepDTAConv1d(lig_vocab_size, channel, ligand_kernel)

        self.proj_p = nn.Linear(self.protein_conv.output_dim, channel)
        self.proj_l = nn.Linear(self.ligand_conv.output_dim, channel)


        self.stereo_conv = SimpleConv1d(stereo_vocab_size, channel, 3)
        #Descriptor MLP branch
        self.descriptor_fc = nn.Sequential(
            nn.Linear(descriptor_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, channel),
            nn.ReLU()
        )

        self.descriptor_bn = nn.BatchNorm1d(channel)

        self.fusion_fc = nn.Sequential(
            nn.Linear(channel*4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, protein, ligand, stereo, descriptor):
        p = F.relu(self.proj_p(self.protein_conv(protein)))
        l = F.relu(self.proj_l(self.ligand_conv(ligand)))

        s = self.stereo_conv(stereo)

        #Descriptor MLP + normalization
        d = F.relu(self.descriptor_fc(descriptor))
        d = self.descriptor_bn(d)

        fused = torch.cat([p, l, s, d], dim=1)
        return self.fusion_fc(fused).squeeze()

