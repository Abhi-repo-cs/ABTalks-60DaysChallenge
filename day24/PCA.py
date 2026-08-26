"""
Day 24 — Principal Component Analysis (PCA)
Run:
    python PCA.py

Creates:
    outputs/pca_2d.png
    outputs/pca_3d.png
    outputs/explained_variance.png
    outputs/cumulative_variance.png
    outputs/model_comparison.csv
    reports/variance_analysis.md
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
REPORTS = BASE / "reports"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def evaluate(pca=None):
    steps = [("scaler", StandardScaler())]
    if pca is not None:
        steps.append(("pca", pca))
    steps.append(("model", LogisticRegression(max_iter=5000, random_state=42)))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    components = X.shape[1] if pca is None else pipe.named_steps["pca"].n_components_
    variance = np.nan if pca is None else pipe.named_steps["pca"].explained_variance_ratio_.sum()
    return components, variance, accuracy_score(y_test, pred), f1_score(y_test, pred)

experiments = [
    ("Baseline (no PCA)", None),
    ("2 components", PCA(n_components=2, random_state=42)),
    ("3 components", PCA(n_components=3, random_state=42)),
    ("90% variance", PCA(n_components=0.90, random_state=42)),
    ("95% variance", PCA(n_components=0.95, random_state=42)),
    ("99% variance", PCA(n_components=0.99, random_state=42)),
]

rows = []
for name, pca in experiments:
    c, v, acc, f1 = evaluate(pca)
    rows.append({
        "Experiment": name,
        "Components": c,
        "Explained_Variance": v,
        "Accuracy": acc,
        "F1_Score": f1
    })
pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)

# Full PCA for variance analysis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_full = PCA().fit(X_scaled)
ratios = pca_full.explained_variance_ratio_
cumulative = np.cumsum(ratios)

# 2D
X2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
s = plt.scatter(X2[:, 0], X2[:, 1], c=y, alpha=0.75)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Projection")
plt.colorbar(s, label="Target class")
plt.tight_layout()
plt.savefig(OUT / "pca_2d.png", dpi=180)
plt.close()

# 3D
X3 = PCA(n_components=3, random_state=42).fit_transform(X_scaled)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
s = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=y, alpha=0.7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title("PCA 3D Projection")
fig.colorbar(s, ax=ax, label="Target class")
plt.tight_layout()
plt.savefig(OUT / "pca_3d.png", dpi=180)
plt.close()

# Variance
plt.figure(figsize=(9, 6))
plt.bar(np.arange(1, len(ratios)+1), ratios)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Principal Component")
plt.tight_layout()
plt.savefig(OUT / "explained_variance.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 6))
plt.plot(np.arange(1, len(cumulative)+1), cumulative, marker="o", markersize=2)
for threshold in [0.90, 0.95, 0.99]:
    k = np.argmax(cumulative >= threshold) + 1
    plt.axhline(threshold, linestyle="--")
    plt.axvline(k, linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.ylim(0, 1.03)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "cumulative_variance.png", dpi=180)
plt.close()

print("PCA experiment completed successfully.")
