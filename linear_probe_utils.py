import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit, softmax
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error


# Precompute the features from the encoder and store them
def precompute_features(encoder, loader, device):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        print("Precomputing features...")
        for wave, label in tqdm(loader):
            bs, _, _ = wave.shape
            print("shape wave ", wave.shape)

            print("mean wave ", wave[0].mean())
            wave = wave.to(device)
            feature = encoder.representation(wave)  # (bs,c*50,384)
            all_features.append(feature.cpu())
            all_labels.append(label)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    return all_features, all_labels


def features_dataloader(encoder, loader, batch_size=32, shuffle=True, device="cpu"):
    features, labels = precompute_features(encoder, loader, device=device)
    """save features and labels """
    torch.save(features, "/home/nadja/ECG_JEPA_Git/features_mvo.pt")
    torch.save(labels, "/home/nadja/ECG_JEPA_Git/labels_mvo.pt")

    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    return dataloader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, apply_bn=False):
        super(LinearClassifier, self).__init__()
        self.apply_bn = apply_bn
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        if self.apply_bn:
            x = self.bn(x)

        x = self.fc(x)
        return x


class FinetuningClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim, num_labels, device="cpu", apply_bn=False):
        super(FinetuningClassifier, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        # self.bn = nn.BatchNorm1d(encoder_dim, affine=False, eps=1e-6) # this outputs nan value in mixed precision
        self.fc = LinearClassifier(encoder_dim, num_labels, apply_bn=apply_bn)

    def forward(self, x):
        bs, _, _ = x.shape
        x = self.encoder.representation(x)
        # x = self.bn(x)
        x = self.fc(x)
        return x


class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim=384):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        return x.view(-1)


def train_multilabel(
    num_epochs,
    linear_model,
    optimizer,
    criterion,
    scheduler,
    train_loader_linear,
    test_loader_linear,
    device,
    print_every=True,
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0

    for epoch in range(num_epochs):
        linear_model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = linear_model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            linear_model.eval()
            for batch_features, batch_labels in test_loader_linear:
                batch_features = batch_features.to(device)
                outputs = linear_model(batch_features)
                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_probs = expit(all_outputs)

        auc_scores = [
            roc_auc_score(all_labels[:, i], all_outputs[:, i])
            if np.unique(all_labels[:, i]).size > 1
            else float("nan")
            for i in range(all_labels.shape[1])
        ]
        avg_auc = np.nanmean(auc_scores)

        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = (all_probs >= 0.5).astype(int)
        macro_f1 = f1_score(all_labels, predicted_labels, average="macro")
        tp = np.sum(all_labels[:, 0] * predicted_labels[:, 0])
        indices_tp = np.where((all_labels[:, 0] * (predicted_labels[:, 0])) == 1)[0]

        tn = np.sum((1 - all_labels[:, 0]) * (1 - predicted_labels[:, 0]))

        fp = np.sum(all_labels[:, 1] * predicted_labels[:, 0])
        fn = np.sum((all_labels[:, 0]) * (1 - predicted_labels[:, 0]))
        indices_fn = np.where((all_labels[:, 0] * (1 - predicted_labels[:, 0])) == 1)[0]

        acc = (tn + tp) / (fp + tp + fn + tn)
        pos = np.sum(all_labels[:, 0])
        if print_every:
            print(
                f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}"
            )

    return avg_auc, macro_f1, predicted_labels, all_labels, tp, pos, tn, fp, fn, acc, indices_tp, indices_fn


def train_multiclass(
    num_epochs,
    model,
    criterion,
    optimizer,
    train_loader_linear,
    test_loader_linear,
    device,
    scheduler=None,
    print_every=False,
    amp=False,
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    macro_f1 = 0.0

    if amp:
        scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                scaler.scale(loss).backward()  # Scale the loss and backpropagate
                scaler.step(optimizer)  # Step the optimizer
                scaler.update()  # Update the scaler

            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            model.eval()
            for minibatch, (batch_features, batch_labels) in enumerate(
                test_loader_linear
            ):
                batch_features = batch_features.to(device)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_features)
                else:
                    outputs = model(batch_features)

                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_outputs = np.vstack(all_outputs)

        if amp:
            all_outputs = np.float32(all_outputs)

        all_probs = softmax(all_outputs, axis=1)

        # Compute ROC AUC score
        avg_auc = roc_auc_score(
            all_labels, all_probs, average="macro", multi_class="ovo"
        )
        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = np.argmax(all_outputs, axis=1)

        macro_f1 = f1_score(all_labels, predicted_labels, average="macro")
        if print_every:
            print(
                f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}"
            )

    print(f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}")
    return avg_auc, macro_f1

### train regression

def train_regression(
    num_epochs,
    model,
    criterion,
    optimizer,
    train_loader_linear,
    test_loader_linear,
    device,
    scheduler=None,
    print_every=False,
    amp=False,
):
    iterations_per_epoch = len(train_loader_linear)
    best_mse = float("inf")  # We want to minimize MSE
    best_r2 = -float("inf")  # R² should be maximized

    if amp:
        scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            optimizer.zero_grad()

            # Mixed precision training
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                scaler.scale(loss).backward()  # Scale the loss and backpropagate
                scaler.step(optimizer)  # Step the optimizer
                scaler.update()  # Update the scaler

            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            model.eval()
            for minibatch, (batch_features, batch_labels) in enumerate(
                test_loader_linear
            ):
                batch_features = batch_features.to(device)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_features)
                else:
                    outputs = model(batch_features)

                print(outputs.shape, flush = True)
                if len(outputs.shape)<2:
                    outputs = outputs.unsqueeze(1)
                  #  batch_labels = batch_labels.unsqueeze(1)

                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)

        if amp:
            all_outputs = np.float32(all_outputs)


        # Compute regression metrics
        print(all_labels.shape, all_outputs.shape, flush = True)
        mse = mean_squared_error(all_labels, all_outputs)
        r2 = r2_score(all_labels, all_outputs)

        # Save best MSE and R² score
        if mse < best_mse:
            best_mse = mse
        if r2 > best_r2:
            best_r2 = r2

        if print_every:
            print(f"Epoch({epoch}) MSE: {mse:.4f} (Best: {best_mse:.4f}), R²: {r2:.4f} (Best: {best_r2:.4f})")


    print(f"Final: MSE: {mse:.4f} (Best: {best_mse:.4f}), R²: {r2:.4f} (Best: {best_r2:.4f})")
    return best_mse, best_r2, all_labels, all_outputs