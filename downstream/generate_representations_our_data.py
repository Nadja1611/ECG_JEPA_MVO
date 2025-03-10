import numpy as np
import torch
import os
import sys, os
import matplotlib.pyplot as plt

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from models import load_encoder
from ecg_data import ECGDataset, ECGDataset_pretrain
from scipy.signal import resample
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Load the encoder
ckpt_dir = "/gpfs/data/fs72515/nadja_g/ECG_JEPA/model_weights/epoch20.pth"  # See https://github.com/sehunfromdaegu/ECG_JEPA for checkpoint download link
encoder, dim = load_encoder(
    ckpt_dir=ckpt_dir
)  # dim is the dimension of the latent space
import numpy as np
from scipy.signal import resample

encoder.eval()
encoder.to("cpu")


# Dummy ECG data
waves = torch.load("/gpfs/data/fs72515/nadja_g/ECG_JEPA/downstream_tasks/ecgs_train_unprocessed.pt")

data = np.array([resample(waves[i], 1250, axis=1) for i in range(len(waves))])
data_new = np.zeros_like(waves)
data_new[:,:6,:1250] = data[:,:6]
data_new[:,6:,1250:] = data[:,6:]
waves = data_new
plt.plot(waves[23][3])
plt.savefig("waves_new.png")
print("shape " + str(waves.shape))
labels = torch.load("/gpfs/data/fs72515/nadja_g/ECG_JEPA/downstream_tasks/mvo_bin_train.pt")
labels = np.array(labels)
# Dataset with labels.
dataset_with_labels = ECGDataset(waves, labels)

all_labels = []
all_features = []
# Create a dataloader
dataloader_with_labels = torch.utils.data.DataLoader(
    dataset_with_labels, batch_size=4, shuffle=True, num_workers=2
)
with torch.no_grad():
    for wave, target in dataloader_with_labels:
        
        repr = encoder.representation(wave.to("cpu"))  # (bs, 8, 2500) -> (bs, dim)
        print("rep ", repr.shape)
        all_features.append(repr.cpu())
        all_labels.append(target)

all_features = torch.cat(all_features)
all_labels = torch.cat(all_labels)
torch.save(all_features, "/gpfs/data/fs72515/nadja_g/ECG_JEPA/embeddings/embeddings_ours.pt")
torch.save(all_labels, "/gpfs/data/fs72515/nadja_g/ECG_JEPA/embeddings/labels_ours.pt")


print(f"Representation shape: {repr.shape}")
