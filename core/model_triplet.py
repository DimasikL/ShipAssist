from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import random


def get_loss_target(problem_mode: str, y: torch.Tensor):
    if problem_mode == 'clf':
        weight = compute_class_weight('balanced', y=y.detach().numpy(), classes=np.unique(y))
        return nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32))
    elif problem_mode == 'reg':
        return nn.MSELoss()
    else:
        raise ValueError(f"mode={problem_mode}")


class TorchClfBase(nn.Module):
    def __init__(
            self,
            embedding_dim: int, n_out: int, device: str, epochs: int, batch_size: int, lr: float,
            norm_x: bool, norm_rows: bool,
            hidden_neurons: int, hidden_layers: int, weight_decay: float, save_best_val: bool, problem_mode: str,
            dropout_rate: float,
            verbose=False
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        assert 0 <= self.dropout_rate <= 0.5, self.dropout_rate

        assert problem_mode in ['reg', 'clf'], problem_mode
        self.problem_mode = problem_mode
        self.norm_x = norm_x
        self.norm_rows = norm_rows

        self.verbose = verbose
        if hidden_layers >= 1:
            assert hidden_layers >= 1
            self.embedder = torch.nn.Sequential(
                nn.Linear(embedding_dim, hidden_neurons),
                nn.ReLU(),
                *[
                    layer
                    for _ in range(hidden_layers - 1)
                    for layer in [
                        nn.Linear(hidden_neurons, hidden_neurons),
                        nn.LayerNorm(hidden_neurons),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout_rate)
                    ]
                ]
            )
            self.last_layer = nn.Sequential(nn.Linear(hidden_neurons, n_out))
        else:
            self.embedder = None
            self.last_layer = nn.Linear(embedding_dim, n_out)

        if problem_mode == 'clf':
            self.le = LabelEncoder()
        else:
            self.le = None

        self.scaler = StandardScaler()

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_neurons = hidden_neurons
        self.save_best_val = save_best_val

        self.epochs = epochs
        self.device = device

    @property
    def classes_(self):
        return self.le.classes_

    def fit_and_loss_on_subset(self, loader, optimizer):
        raise Exception("Not implemented")

    def verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def fit_n_epochs(self, optimizer, train_loader, val_loader):
        best_loss = 1e20
        best_model = deepcopy(self.last_layer.state_dict())

        for epoch in range(self.epochs):
            train_loss = self.fit_and_loss_on_subset(loader=train_loader, optimizer=optimizer)
            val_loss = self.fit_and_loss_on_subset(loader=val_loader, optimizer=None)

            self.verbose_print(
                f"[Triplet] Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_loss:
                best_model = deepcopy(self.last_layer.state_dict())
                best_loss = val_loss
                self.verbose_print('best model updated')

        if self.save_best_val:
            self.last_layer.load_state_dict(best_model)

    def get_emb(self, x):
        x = self.embedder(x)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x):
        if self.norm_rows:
            x = F.normalize(x, p=2, dim=1)

        if self.embedder is not None:
            x = self.get_emb(x)
        # else:
        #     x = F.normalize(x, p=2, dim=1)

        return self.last_layer(x)

    def scale_f(self, X):
        if self.norm_x:
            return self.scaler.transform(X)
        else:
            return X

    def predict_logits(self, X) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            if isinstance(X, pd.DataFrame): X = X.to_numpy()
            if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.forward(X)

            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            else:
                pass
            return logits

    def predict_proba(self, X):
        return torch.nn.functional.softmax(self.predict_logits(X), dim=1).detach().cpu().numpy()

    def predict(self, X):
        X = self.scale_f(X)
        if self.problem_mode == 'clf':
            return self.le.inverse_transform(self.predict_proba(X).argmax(axis=-1).flatten())
        elif self.problem_mode == 'reg':
            return self.predict_logits(X).detach().cpu().numpy()
        else:
            raise ValueError(f"{self.problem_mode}")


class LinearOrMlpModel(TorchClfBase):
    def __init__(self, x_val, y_val, **kwargs):
        super().__init__(**kwargs)
        self.x_val = x_val
        self.y_val = y_val

    def fit_and_loss_on_subset(self, loader, optimizer):
        training = optimizer is not None
        if training:
            self.train()
        else:
            self.eval()

        total_loss = 0
        with torch.set_grad_enabled(training):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self(xb)
                loss = self.loss_target(logits, yb)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, X, y):
        self.to(self.device)
        self.train()

        if self.problem_mode == 'clf':
            y_dtype = torch.long
            self.le = self.le.fit(y)
            y = self.le.transform(y)
            self.y_val = self.le.transform(self.y_val)
        elif self.problem_mode == 'reg':
            y_dtype = torch.float32
        else:
            raise ValueError(f"{self.problem_mode}")

        self.scaler.fit(X)

        X = torch.tensor(self.scale_f(X), dtype=torch.float32)
        y = torch.tensor(y, dtype=y_dtype)
        self.x_val = torch.tensor(self.scale_f(self.x_val), dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=y_dtype)
        self.loss_target = get_loss_target(problem_mode=self.problem_mode, y=y)

        train_dataset = TensorDataset(X, y)
        val_dataset = TensorDataset(self.x_val, self.y_val)

        self.fit_n_epochs(
            optimizer=torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            train_loader=DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            val_loader=DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        )


class LinearClfReg(TorchClfBase):
    def __init__(self, x_val, y_val, **kwargs):
        super().__init__(**kwargs)
        self.x_val = x_val
        self.y_val = y_val

    def fit_and_loss_on_subset(self, loader, optimizer):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self(xb)
            loss = self.loss_target(logits, yb)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        return total_loss / len(loader)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label: 1 - похожие, 0 - разные
        distances = F.pairwise_distance(output1, output2)
        loss = label * torch.pow(distances, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        return F.relu(d_pos - d_neg + self.margin).mean()


class MultiTaskWDomainTransfer(TorchClfBase):
    def __init__(
            self,
            x_val1, y_val1, x_val2, y_val2,
            margin: float, alpha: float,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert self.problem_mode == 'clf', self.problem_mode
        self.x_val1 = x_val1
        self.y_val1 = y_val1
        self.x_val2 = x_val2
        self.y_val2 = y_val2
        self.alpha = alpha
        self.margin = margin

        if self.problem_mode == 'clf':
            self.le = LabelEncoder().fit(self.y_val1)
        else:
            self.le = None

    def get_domain_transfer_dset(self, x1, x2, y1, y2) -> TensorDataset:
        raise Exception("Not implemented")

    def fit(self, x_train1, y_train1, x_train2, y_train2):
        self.to(self.device)
        self.train()

        le = LabelEncoder()
        x_train1 = np.array(x_train1)
        y_train1 = np.array(le.fit_transform(y_train1))

        x_train2 = np.array(x_train2)
        y_train2 = np.array(le.fit_transform(y_train2))

        self.x_val1 = np.array(self.x_val1)
        self.y_val1 = np.array(le.fit_transform(self.y_val1))

        self.x_val2 = np.array(self.x_val2)
        self.y_val2 = np.array(le.fit_transform(self.y_val2))

        self.scaler.fit(np.concatenate([x_train1, x_train2]))
        x_train1 = self.scaler.transform(x_train1)
        x_train2 = self.scaler.transform(x_train2)
        self.x_val1 = self.scaler.transform(self.x_val1)
        self.x_val2 = self.scaler.transform(self.x_val2)

        train_dataset = self.get_domain_transfer_dset(x1=x_train1, x2=x_train2, y1=y_train1, y2=y_train2)
        val_dataset = self.get_domain_transfer_dset(x1=self.x_val1, x2=self.x_val2, y1=self.y_val1, y2=self.y_val2)

        self.fit_n_epochs(
            optimizer=torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            train_loader=DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            val_loader=DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        )


class LinearContrastive(MultiTaskWDomainTransfer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_loss = ContrastiveLoss(margin=kwargs['margin'])

    def get_domain_transfer_dset(self, x1, x2, y1, y2) -> TensorDataset:
        x1_list, x2_list, cls_labels, sim_labels = [], [], [], []

        for i in range(len(x1)):
            anchor = x1[i]
            label = y1[i]

            # Положительная пара: с тем же классом из x_val
            pos_indices = np.where(y2 == label)[0]
            if len(pos_indices) > 0:
                pos_idx = random.choice(pos_indices)
                x1_list.append(anchor)
                x2_list.append(x2[pos_idx])
                cls_labels.append(label)
                sim_labels.append(1)

            # Отрицательная пара: с другим классом
            neg_indices = np.where(y2 != label)[0]
            if len(neg_indices) > 0:
                neg_idx = random.choice(neg_indices)
                x1_list.append(anchor)
                x2_list.append(x2[neg_idx])
                cls_labels.append(label)
                sim_labels.append(0)

        x1 = torch.tensor(np.array(x1_list), dtype=torch.float32)
        x2 = torch.tensor(np.array(x2_list), dtype=torch.float32)
        cls_labels = torch.tensor(np.array(cls_labels), dtype=torch.long)
        sim_labels = torch.tensor(np.array(sim_labels), dtype=torch.float32)
        return TensorDataset(x1, x2, cls_labels, sim_labels)

    def fit_and_loss_on_subset(self, optimizer, loader):
        total_loss = 0
        for xb1, xb2, yb_cls, yb_sim in loader:
            xb1, xb2 = xb1.to(self.device), xb2.to(self.device)
            yb_cls, yb_sim = yb_cls.to(self.device), yb_sim.to(self.device)

            out1 = self.get_emb(xb1)
            out2 = self.get_emb(xb2)

            loss_target = self.loss_target(self.last_layer(out1), yb_cls)
            loss_contrast = self.contrastive_loss(out1, out2, yb_sim)
            loss = loss_target + self.alpha * loss_contrast

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)


class LinearTriplet(MultiTaskWDomainTransfer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.triplet_loss = TripletLoss(margin=kwargs['margin'])

    def get_domain_transfer_dset(self, x1, x2, y1, y2) -> TensorDataset:
        anchor_list, pos_list, neg_list, cls_labels = [], [], [], []

        class_to_indices = {cls: np.where(y1 == cls)[0] for cls in np.unique(y1)}

        for idx, anchor_vec in enumerate(x1):
            label = y1[idx]

            # Положительные примеры
            pos_candidates = class_to_indices[label]
            pos_candidates = pos_candidates[pos_candidates != idx]
            if len(pos_candidates) == 0:
                continue
            pos_idx = random.choice(pos_candidates)
            pos_vec = x1[pos_idx]

            # Отрицательные примеры
            neg_labels = [l for l in class_to_indices if l != label]
            neg_label = random.choice(neg_labels)
            neg_idx = random.choice(class_to_indices[neg_label])
            neg_vec = x1[neg_idx]

            # Добавление в список
            anchor_list.append(anchor_vec)
            pos_list.append(pos_vec)
            neg_list.append(neg_vec)
            cls_labels.append(label)  # классифицируем только anchor

        # Преобразование в тензоры
        anchor = torch.tensor(np.array(anchor_list), dtype=torch.float32)
        pos = torch.tensor(np.array(pos_list), dtype=torch.float32)
        neg = torch.tensor(np.array(neg_list), dtype=torch.float32)
        cls_labels = torch.tensor(np.array(cls_labels), dtype=torch.long)

        return TensorDataset(anchor, pos, neg, cls_labels)

    def fit_and_loss_on_subset(self, loader, optimizer):
        total_loss = 0
        for a, p, n, yb in loader:
            a, p, n, yb = a.to(self.device), p.to(self.device), n.to(self.device), yb.to(self.device)

            out_a = self.get_emb(a)
            out_p = self.get_emb(p)
            out_n = self.get_emb(n)

            loss_target = self.loss_target(self.last_layer(out_a), yb)
            loss_triplet = self.triplet_loss(out_a, out_p, out_n)
            loss = loss_target + self.alpha * loss_triplet

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        return total_loss / len(loader)
