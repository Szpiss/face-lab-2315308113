from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


sns.set_theme(style="whitegrid")


@dataclass
class DatasetBundle:
    raw_images: np.ndarray
    flat_images: np.ndarray
    labels: np.ndarray
    image_paths: list[str]


class FaceDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, image_size: int = 128):
        transform_ops = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ]
        self.transform = transforms.Compose(transform_ops)
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_face_dataset(data_dir: str | Path) -> DatasetBundle:
    base = Path(data_dir)
    raw_images: list[np.ndarray] = []
    flat_images: list[np.ndarray] = []
    labels: list[int] = []
    image_paths: list[str] = []

    for person_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name):
        for image_path in sorted(person_dir.glob("*.jpg")):
            image = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
            raw_images.append(image)
            flat_images.append(image.reshape(-1))
            labels.append(int(person_dir.name))
            image_paths.append(str(image_path))

    return DatasetBundle(
        raw_images=np.stack(raw_images),
        flat_images=np.stack(flat_images),
        labels=np.array(labels),
        image_paths=image_paths,
    )


def plot_subject_samples(bundle: DatasetBundle, save_path: str | Path, num_subjects: int = 5) -> None:
    unique_labels = sorted(np.unique(bundle.labels))[:num_subjects]
    fig, axes = plt.subplots(1, num_subjects, figsize=(14, 3))
    for ax, subject in zip(axes, unique_labels):
        idx = int(np.where(bundle.labels == subject)[0][0])
        ax.imshow(bundle.raw_images[idx], cmap="gray")
        ax.set_title(f"ID {subject}")
        ax.axis("off")
    fig.suptitle("前五名受试者的第一个样本", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_pca(
    flat_images: np.ndarray,
    seed: int,
    variance_threshold: float,
    save_path: str | Path,
) -> tuple[PCA, np.ndarray, dict[str, float | int]]:
    pca_full = PCA(random_state=seed)
    pca_full.fit(flat_images)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)

    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(flat_images)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative, color="#2e6f95", linewidth=2)
    ax.axhline(variance_threshold, color="#c1121f", linestyle="--", label=f"{variance_threshold:.0%}阈值")
    ax.axvline(n_components, color="#e36414", linestyle="--", label=f"n_components={n_components}")
    ax.set_xlabel("主成分个数")
    ax.set_ylabel("累计解释方差比")
    ax.set_title("PCA累计解释方差曲线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    info = {
        "n_components": n_components,
        "variance_threshold": variance_threshold,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return pca, reduced, info


def plot_k_distance(reduced: np.ndarray, save_path: str | Path, k: int = 5) -> dict[str, list[float]]:
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(reduced)
    distances, _ = neighbors.kneighbors(reduced)
    sorted_distances = np.sort(distances[:, -1])

    percentiles = np.percentile(sorted_distances, [80, 90, 95, 97, 99]).round(4).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_distances, color="#6a994e", linewidth=1.8)
    ax.set_title(f"k距离曲线（k={k}）")
    ax.set_xlabel("样本排序后索引")
    ax.set_ylabel(f"第{k}近邻距离")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"percentiles": percentiles}


def run_dbscan(
    reduced: np.ndarray,
    raw_images: np.ndarray,
    image_paths: list[str],
    eps: float,
    min_samples: int,
    save_path: str | Path,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(reduced)
    noise_indices = np.where(labels == -1)[0][:5]
    normal_indices = np.where(labels != -1)[0][:5]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, idx in enumerate(noise_indices):
        axes[0, i].imshow(raw_images[idx], cmap="gray")
        axes[0, i].set_title(f"噪声 {Path(image_paths[idx]).stem}")
        axes[0, i].axis("off")
    for j in range(len(noise_indices), 5):
        axes[0, j].axis("off")

    for i, idx in enumerate(normal_indices):
        axes[1, i].imshow(raw_images[idx], cmap="gray")
        axes[1, i].set_title(f"正常 {Path(image_paths[idx]).stem}")
        axes[1, i].axis("off")
    for j in range(len(normal_indices), 5):
        axes[1, j].axis("off")

    fig.suptitle("DBSCAN去噪结果示例", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    info = {
        "noise_count": int((labels == -1).sum()),
        "normal_count": int((labels != -1).sum()),
    }
    return labels, info


def prepare_classical_data(
    reduced: np.ndarray,
    labels: np.ndarray,
    noise_labels: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    keep_mask = noise_labels != -1
    reduced_clean = reduced[keep_mask]
    labels_clean = labels[keep_mask]

    scaler = StandardScaler()
    reduced_scaled = scaler.fit_transform(reduced_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        reduced_scaled,
        labels_clean,
        test_size=0.3,
        random_state=seed,
        stratify=labels_clean,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "keep_mask": keep_mask,
        "scaler": scaler,
    }


def _micro_auc_score(y_true: np.ndarray, y_score: np.ndarray, class_ids: list[int]) -> float:
    y_bin = label_binarize(y_true, classes=class_ids)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    return float(auc(fpr, tpr))


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    max_classes: int = 20,
) -> None:
    labels_subset = sorted(np.unique(y_true))[:max_classes]
    subset_mask = np.isin(y_true, labels_subset)
    cm = confusion_matrix(y_true[subset_mask], y_pred[subset_mask], labels=labels_subset)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=False)
    ax.set_title("混淆矩阵（前20个类别，便于阅读）")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks(np.arange(len(labels_subset)) + 0.5)
    ax.set_yticks(np.arange(len(labels_subset)) + 0.5)
    ax.set_xticklabels(labels_subset, rotation=90, fontsize=8)
    ax.set_yticklabels(labels_subset, rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_micro_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_ids: list[int],
    save_path: str | Path,
) -> float:
    y_bin = label_binarize(y_true, classes=class_ids)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#d62828", linewidth=2, label=f"micro-average AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC曲线（micro-average）")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    class_ids: list[int],
    save_dir: str | Path,
) -> dict[str, float | str]:
    save_base = Path(save_dir)
    ensure_dir(save_base)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc_val = _save_micro_roc(
        y_true,
        y_score,
        class_ids,
        save_base / f"{model_name}_roc.png",
    )
    _save_confusion_matrix(
        y_true,
        y_pred,
        save_base / f"{model_name}_cm.png",
    )

    return {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(roc_auc_val),
    }


def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str | Path,
    seed: int,
) -> tuple[GaussianNB, dict[str, float | str], dict[str, float]]:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        GaussianNB(),
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7]},
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model: GaussianNB = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    metrics = evaluate_model(
        "naive_bayes",
        y_test,
        y_pred,
        y_score,
        sorted(np.unique(y_train)),
        save_dir,
    )
    return best_model, metrics, grid.best_params_


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str | Path,
    seed: int,
) -> tuple[SVC, dict[str, float | str], dict[str, float]]:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(probability=True, random_state=seed),
        param_grid={
            "C": [3, 5],
            "kernel": ["rbf"],
            "gamma": ["scale", 0.01],
        },
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model: SVC = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    metrics = evaluate_model(
        "svm",
        y_test,
        y_pred,
        y_score,
        sorted(np.unique(y_train)),
        save_dir,
    )
    return best_model, metrics, grid.best_params_


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str | Path,
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, float | str], dict[str, float]]:
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        param_grid={
            "n_estimators": [150, 200],
            "max_depth": [None, 20],
        },
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model: RandomForestClassifier = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    metrics = evaluate_model(
        "random_forest",
        y_test,
        y_pred,
        y_score,
        sorted(np.unique(y_train)),
        save_dir,
    )
    return best_model, metrics, grid.best_params_


def train_cnn(
    raw_images: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    save_dir: str | Path,
    seed: int,
    epochs: int = 8,
    batch_size: int = 64,
) -> tuple[SmallCNN, dict[str, float | str], list[float], dict[str, int]]:
    device = get_device()
    raw_clean = raw_images[keep_mask]
    labels_clean = labels[keep_mask]
    class_ids = sorted(np.unique(labels_clean))
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
    label_idx = np.array([class_to_idx[item] for item in labels_clean])

    X_train, X_test, y_train, y_test = train_test_split(
        raw_clean,
        label_idx,
        test_size=0.3,
        random_state=seed,
        stratify=label_idx,
    )

    train_loader = DataLoader(FaceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(FaceDataset(X_test, y_test), batch_size=batch_size * 2, shuffle=False)

    model = SmallCNN(num_classes=len(class_ids)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        history.append(epoch_loss / len(train_loader.dataset))

    model.eval()
    y_pred_list: list[int] = []
    y_score_list: list[np.ndarray] = []
    y_true_list: list[int] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_pred_list.extend(preds.tolist())
            y_true_list.extend(yb.numpy().tolist())
            y_score_list.append(probs)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_score = np.vstack(y_score_list)

    save_base = Path(save_dir)
    ensure_dir(save_base)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, epochs + 1), history, marker="o", color="#7b2cbf")
    ax.set_title("CNN训练损失曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(save_base / "cnn_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    metrics = evaluate_model("cnn", y_true, y_pred, y_score, list(range(len(class_ids))), save_base)
    extra = {"device": str(device), "epochs": epochs, "batch_size": batch_size}
    return model, metrics, history, extra


def save_metrics_table(metrics_list: list[dict[str, float | str]], save_dir: str | Path) -> pd.DataFrame:
    save_base = ensure_dir(Path(save_dir))
    df = pd.DataFrame(metrics_list)
    df.to_csv(save_base / "metrics_summary.csv", index=False)
    with (save_base / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_list, file, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    display_df = df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    fig.tight_layout()
    fig.savefig(save_base / "metrics_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def save_metrics_bar_chart(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    display_df = metrics_df.copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    indices = np.arange(len(display_df))
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors = ["#2a9d8f", "#457b9d", "#f4a261", "#e76f51"]

    for idx, metric_name in enumerate(metric_names):
        ax.bar(
            indices + idx * width,
            display_df[metric_name],
            width=width,
            label=metric_name,
            color=colors[idx],
        )

    ax.set_xticks(indices + 1.5 * width)
    ax.set_xticklabels(display_df["model"])
    ax.set_ylim(0, 0.5)
    ax.set_title("四种模型主要指标对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

