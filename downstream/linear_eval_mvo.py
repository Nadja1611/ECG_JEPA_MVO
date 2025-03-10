import yaml
import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module

# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from ecg_data_500Hz import *

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
from boxplots import boxplot

def parse():
    parser = argparse.ArgumentParser("ECG downstream training")

    # parser.add_argument('--model_name',
    #                     default="ejepa_random",
    #                     type=str,
    #                     help='resume from checkpoint')

    parser.add_argument(
        "--ckpt_dir",
        default="../weights/multiblock_epoch100.pth",
        type=str,
        metavar="PATH",
        help="pretrained encoder checkpoint",
    )

    parser.add_argument(
        "--output_dir",
        default="./output/linear_eval",
        type=str,
        metavar="PATH",
        help="output directory",
    )

    parser.add_argument(
        "--data_mvo",
        default="",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset mvo directory",
    )

    parser.add_argument("--dataset", default="ptbxl", type=str, help="dataset name")

    parser.add_argument(
        "--data_dir",
        default="/mount/ecg/ptb-xl-1.0.3/",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset directory",
    )

    parser.add_argument(
        "--task", default="multilabel", type=str, help="downstream task"
    )

    parser.add_argument(
        "--data_percentage",
        default=1.0,
        type=float,
        help="data percentage (from 0 to 1) to use in few-shot learning",
    )

    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()

    with open(
        os.path.realpath(f"../configs/downstream/linear_eval/linear_eval_ejepa.yaml"),
        "r",
    ) as f:
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


    volumes = torch.load(os.path.join(data_path, 'mvo_volume_experts_val.pt'))

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

    data_percentage = config["data_percentage"]
    n_trial = 1 if data_percentage == 1 else 3

    AUCs, F1s, Conf, MSEs, R2s = [], [], [], [], []
    logging.info(f"Start training...")
    print(f"Start training...")
    for n in range(n_trial):
        num_samples = len(waves_train)
        if data_percentage < 1.0:
            num_desired_samples = round(num_samples * data_percentage)
            selected_indices = np.random.choice(
                num_samples, num_desired_samples, replace=False
            )
            waves_train_selected = waves_train[selected_indices]
            labels_train_selected = labels_train[selected_indices]
        else:
            waves_train_selected = waves_train
            labels_train_selected = labels_train

        num_workers = config["dataloader"]["num_workers"]
        waves_train = np.concatenate(
            (waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1
        )
        waves_test = np.concatenate(
            (waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1
        )

        train_dataset = ECGDataset(waves_train, labels_train)
        test_dataset = ECGDataset(waves_test, labels_test)
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=num_workers
        )

        bs = config["dataloader"]["batch_size"]
        train_loader_linear = features_dataloader(
            encoder, train_loader, batch_size=bs, shuffle=True, device=device
        )
        test_loader_linear = features_dataloader(
            encoder, test_loader, batch_size=bs, shuffle=False, device=device
        )

        num_epochs = config["train"]["epochs"]
        lr = config["train"]["lr"]
        
        if config['task'] == 'multilabel' or config['task'] == 'multiclass':
            criterion = (
                nn.BCEWithLogitsLoss()
                if config["task"] == "multilabel"
                else nn.CrossEntropyLoss()
            )
            linear_model = LinearClassifier(embed_dim, n_labels).to(device)

        if 'regression' in config['task']:
            ''' no matter if mvo or volume of scar regression task'''
            criterion = nn.MSELoss()
            linear_model = SimpleLinearRegression(embed_dim).to(device)

        optimizer = optim.AdamW(linear_model.parameters(), lr=lr)
        iterations_per_epoch = len(train_loader_linear)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs * iterations_per_epoch,
            cycle_mul=1,
            lr_min=lr * 0.1,
            cycle_decay=0.1,
            warmup_lr_init=lr * 0.1,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True,
        )
        if config["task"] == "multilabel" or config["task"] == "multiclass":
            if config["task"] == "multilabel":
                auc, f1, predicted_labels, all_labels, tp, pos, tn, fp, fn, acc, indices_tp, indices_fn = (
                    train_multilabel(
                        num_epochs,
                        linear_model,
                        optimizer,
                        criterion,
                        scheduler,
                        train_loader_linear,
                        test_loader_linear,
                        device,
                        print_every=True,
                    )
                )
            else:
                auc, f1 = train_multiclass(
                    num_epochs,
                    linear_model,
                    criterion,
                    optimizer,
                    train_loader_linear,
                    test_loader_linear,
                    device,
                    scheduler=scheduler,
                    print_every=True,
                    amp=False,
                )

            AUCs.append(auc)
            F1s.append(f1)
            logging.info(f"Trial {n + 1}: AUC: {auc:.3f}, F1: {f1:.3f}, Acc: {acc:.3f}")
            print(f"Trial {n + 1}: AUC: {auc:.3f}, F1: {f1:.3f}")
            print("GT:  ", all_labels)
            print("predictions:", predicted_labels)
            print("TP:", tp)
            print("Pos:", pos)
            print("TN:", tn)
            print("FP:", fp)
            print("ACC:", acc)

    
        
        if 'regression' in config['task']:
            mse, r2, all_labels, predicted_labels = train_regression(
                    num_epochs,
                    linear_model,
                    criterion,
                    optimizer,
                    train_loader_linear,
                    test_loader_linear,
                    device,
                    scheduler=scheduler,
                    print_every=True,
                    amp=False,
                )

            MSEs.append(mse)
            R2s.append(r2)
            logging.info(f"Trial {n + 1}: MSE: {mse:.3f}, R2: {r2:.3f}")
            print("GT:  ", all_labels)
            print("predictions:", predicted_labels)
            
    if 'regression' in config['task']:
        mean_mse = np.mean(MSEs)
        std_mse = np.std(MSEs)
        mean_r2 = np.mean(R2s)
        std_r2 = np.std(R2s)
        logging.info(
            f"Mean MSE: {mean_mse:.3f} +- {std_mse:.3f}, Mean R2: {mean_r2:.3f} +- {std_r2:.3f}"
        )

    if config['task'] == 'multilabel' or config['task'] == 'multiclass':
        mean_auc = np.mean(AUCs)
        std_auc = np.std(AUCs)
        mean_f1 = np.mean(F1s)
        std_f1 = np.std(F1s)
        logging.info(
            f"Mean AUC: {mean_auc:.3f} +- {std_auc:.3f}, Mean F1: {mean_f1:.3f} +- {std_f1:.3f}"
        )
        print(
            f"Mean AUC: {mean_auc:.3f} +- {std_auc:.3f}, Mean F1: {mean_f1:.3f} +- {std_f1:.3f}"
        )
        boxplot(indices_tp, indices_fn, volumes)    

if __name__ == "__main__":
    config = parse()

    # pretrained_ckpt_dir = {
    #     'ejepa_random': f"../weights/randomblock_epoch100.pth",
    #     'ejepa_multiblock': f"../weights/multiblock_epoch100.pth",
    #     # 'cmsc': "../weights/shao+code15/CMSC/epoch300.pth",
    #     # 'cpc': "../weights/shao+code15/cpc/base_epoch100.pth",
    #     # 'simclr': "../weights/shao+code15/SimCLR/epoch300.pth",
    #     # 'st_mem': "../weights/shao+code15/st_mem/st_mem_vit_base.pth",
    # }

    # config['ckpt_dir'] = pretrained_ckpt_dir[config['model_name']]

    main(config)
