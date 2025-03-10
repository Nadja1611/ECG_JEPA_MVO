import yaml
import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module
import torch
# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from timm.scheduler import CosineLRScheduler
from ecg_data_500Hz import ECGDataset
from torch.utils.data import DataLoader
from models import load_encoder
from linear_probe_utils import (
    features_dataloader,
    train_multilabel,
    train_multiclass,
    train_regression,
    LinearClassifier,
    SimpleLinearRegression
)
import torch.optim as optim
import torch.nn as nn

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from ecg_data_500Hz_ptbxl_ours import *


def parse():
    parser = argparse.ArgumentParser('ECG downstream training')

    # parser.add_argument('--model_name',
    #                     default="mvt_larger_larger",
    #                     type=str,
    #                     help='resume from checkpoint')
    
    parser.add_argument('--ckpt_dir',
                        default="../weights/multiblock_epoch100.pth",
                        type=str,
                        metavar='PATH',
                        help='pretrained encoder checkpoint')
    
    parser.add_argument('--output_dir',
                        default="./output/finetuning",
                        type=str,
                        metavar='PATH',
                        help='output directory')
    
    parser.add_argument('--dataset',
                        default="ptbxl",
                        type=str,
                        help='dataset name')
    
    parser.add_argument('--data_dir',
                        default="/mount/ecg/ptb-xl-1.0.3/",
                        type=str,
                        help='dataset directory')
    
    parser.add_argument('--task',
                        default="multiclass",
                        type=str,
                        help='downstream task')
    
    parser.add_argument('--pathology',
                        default="mvo",
                        type=str,
                        help='medical task to be solved')

    parser.add_argument('--data_percentage',
                        default=1.0,
                        type=float,
                        help='data percentage (from 0 to 1) to use in few-shot learning')
    
    parser.add_argument('--data_mvo',
                        default="", # "/mount/ecg/cpsc_2018/"
                        type=str,
                        help='dataset mvo directory')

    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()


    with open(os.path.realpath(f'../configs/downstream/finetuning/fine_tuning_ejepa.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log filename with current time
    ckpt_name = os.path.splitext(os.path.basename(config["ckpt_dir"]))[0]
    log_filename = os.path.join(
        config["output_dir"],
        f"log_{ckpt_name}_{config['task']}_{config['dataset']}_{current_time}.txt",
    )

    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Log the config dictionary
    logging.info("Configuration:")
    logging.info(yaml.dump(config, default_flow_style=False))

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = config["data_mvo"]
    logging.info(f"Loading {config['dataset']} dataset...")
    print(f"Loading {config['dataset']} dataset...")
    waves_train = torch.load(data_path + "/ecgs_train.pt")
    waves_test = torch.load(data_path + "/ecgs_val.pt")

    if config["task"] == "multilabel" or config["task"] == "multiclass":
        labels_train = torch.load(data_path + "/mvo_bin_train.pt")
        labels_test = torch.load(data_path + "/mvo_bin_val.pt")
    elif config['task'] == "regression_infarct":
        labels_train = torch.load(data_path + "/scarvol_train.pt")
        labels_test = torch.load(data_path + "/scarvol_val.pt")
        labels_test = labels_test/torch.max(labels_test)
    else:
        labels_train = torch.load(data_path + "/mvo_vol_CNN_train.pt")
        labels_test = torch.load(data_path + "/mvo_vol_CNN_val.pt")
        labels_test = labels_test/torch.max(labels_test)


    waves_train  = np.concatenate((waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1)
    waves_test  = np.concatenate((waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1)


    print(labels_train[:4])
    # waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)

    if config["task"] == "multilabel":
        _, n_labels = labels_train.shape
    elif config["task"] == "multiclass":
        n_labels = len(np.unique(labels_train))
    else:
        ''' for the regression case, the output should be one number'''
        n_labels = 1


    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Loading encoder from {config['ckpt_dir']}...")
    print(f"Loading encoder from {config['ckpt_dir']}...")
    encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"])
    encoder = encoder.to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    waves = np.concatenate((waves_train, waves_test), 0)
    labels = np.concatenate((labels_train, labels_test), 0)
    print('shapes ', waves.shape, labels.shape)
    dataset_with_labels = ECGDataset(waves, labels)


    # PTBXL
    waves_p = waves_ptbxl('/gpfs/data/fs72515/nadja_g/ECG_JEPA/physionet/files/ptb-xl/1.0.3/')
    print(f"PTBXL waves shape: {waves_p.shape}")
    logging.info(f"PTBXL waves shape: {waves_p.shape}")
    dataset_ptbxl = ECGDataset_pretrain(waves_p)


    all_labels = []
    all_features = []
    # Create a dataloader
    dataloader_with_labels = torch.utils.data.DataLoader(
        dataset_with_labels, batch_size=128, shuffle=True, num_workers=2
    )
    with torch.no_grad():
        for wave, target in dataloader_with_labels:
            
            repr = encoder.representation(wave.to(device))  # (bs, 8, 2500) -> (bs, dim)
            print("rep ", repr.shape)
            all_features.append(repr.cpu())
            all_labels.append(target)
    with torch.no_grad():
        for wave, target in dataset_ptbxl:
            repr = encoder.representation(wave.to(device))  # (bs, 8, 2500) -> (bs, dim)
            print("rep ", repr.shape)
            all_features.append(repr.cpu())
            all_labels.append(target)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    torch.save(all_features, "/home/nadja/ECG_JEPA_Git/embeddings/embeddings_ours" + config['pathology'] + ".pt")
    torch.save(all_labels, "/home/nadja/ECG_JEPA_Git/embeddings/labels_ours" + config['pathology'] + ".pt")

    print('waves mean: ', wave[0][0].mean())
    print(f"Representation shape: {repr.shape}")      
    LAB = torch.zeros((all_labels.shape[0], 2))  
    LAB[:len(dataset), :2] = all_labels[:len(dataset)] 
    LAB[len(dataset):, :2] = all_labels[len(dataset):] 

    LAB = np.argmax(LAB, axis=1)

    EMB = all_features
    print('emb ' + str(EMB.shape))
    # Choose a method for dimensionality reduction
    method = "tsne"  # Options: "pca", "tsne", "umap"

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=5,  random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)

    # Reduce dimensions
    EMB_2D = reducer.fit_transform(EMB.detach().cpu())

    # Plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("hsv", 7)  # 7-class color palette
    sns.scatterplot(x=EMB_2D[:, 0], y=EMB_2D[:, 1], hue=LAB, palette=palette, alpha=0.7, edgecolor="k")

    plt.title(f"2D Visualization using {method.upper()}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Class", loc="best")
    plt.savefig("/home/nadja/ECG_JEPA_Git/embeddings/embedding_space_tsne_" + config['pathology'] + ".png")

if __name__ == '__main__':
    config = parse()


    # pretrained_ckpt_dir = {
    #     'ejepa_random': f"../weights/random_epoch100.pth",
    #     'ejepa_multiblock': f"../weights/multiblock_epoch100.pth",
    #     # 'cmsc': "../weights/shao+code15/CMSC/epoch300.pth",
    #     # 'cpc': "../weights/shao+code15/cpc/base_epoch100.pth",
    #     # 'simclr': "../weights/shao+code15/SimCLR/epoch300.pth",
    #     # 'st_mem': "../weights/shao+code15/st_mem/st_mem_vit_base.pth",
    # }
        

    # # pretrained_ckpt_dir['mvt_larger_larger'] = f"../weights/shao+code15/block_masking/jepa_v4_20240720_215455_(0.175, 0.225)/epoch{100}.pth"

    # config['ckpt_dir'] = pretrained_ckpt_dir[config['model_name']]

    main(config)