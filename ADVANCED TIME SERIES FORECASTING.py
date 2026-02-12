# advanced_time_series_project.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import optuna
import shap
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# DATA GENERATION (Complex Multivariate Dataset)
# ============================================================

np.random.seed(42)
n = 2000
t = np.arange(n)

seasonal_daily = 10 * np.sin(2 * np.pi * t / 24)
seasonal_weekly = 5 * np.sin(2 * np.pi * t / 168)
trend = 0.005 * t
noise = np.random.normal(scale=2, size=n)

exog1 = np.sin(2 * np.pi * t / 12)
exog2 = np.cos(2 * np.pi * t / 48)
exog3 = np.random.normal(scale=1, size=n)

target = seasonal_daily + seasonal_weekly + trend + 2*exog1 - 1.5*exog2 + noise

df = pd.DataFrame({
    "target": target,
    "exog1": exog1,
    "exog2": exog2,
    "exog3": exog3
})

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

train_size = int(len(df)*0.7)
val_size = int(len(df)*0.15)

train_df = df[:train_size]
val_df = df[train_size:train_size+val_size]
test_df = df[train_size+val_size:]

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

# ============================================================
# DATASET CLASS
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window, horizon):
        self.X = []
        self.y = []

        for i in range(len(data) - window - horizon):
            self.X.append(data[i:i+window])
            self.y.append(data[i+window:i+window+horizon, 0])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODEL
# ============================================================

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        return self.fc(x)

class NBeats(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, dropout):
        super().__init__()
        self.block1 = NBeatsBlock(input_size, hidden_size, horizon, dropout)
        self.block2 = NBeatsBlock(input_size, hidden_size, horizon, dropout)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.block1(x) + self.block2(x)

# ============================================================
# METRICS
# ============================================================

def mase(y_true, y_pred, y_train):
    naive = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def objective(trial):

    window = trial.suggest_int("window", 24, 72)
    hidden = trial.suggest_int("hidden", 128, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    horizon = 24

    train_dataset = TimeSeriesDataset(train_scaled, window, horizon)
    val_dataset = TimeSeriesDataset(val_scaled, window, horizon)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_size = window * df.shape[1]
    model = NBeats(input_size, hidden, horizon, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, actual = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            preds.append(model(X).cpu().numpy())
            actual.append(y.numpy())

    preds = np.vstack(preds)
    actual = np.vstack(actual)

    return np.sqrt(mean_squared_error(actual.flatten(), preds.flatten()))

print("Running Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best Params:", study.best_params)

# ============================================================
# FINAL TRAINING
# ============================================================

params = study.best_params
window = params["window"]
hidden = params["hidden"]
dropout = params["dropout"]
lr = params["lr"]
horizon = 24

train_dataset = TimeSeriesDataset(train_scaled, window, horizon)
test_dataset = TimeSeriesDataset(test_scaled, window, horizon)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = NBeats(window*df.shape[1], hidden, horizon, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
preds, actual = [], []

with torch.no_grad():
    for X, y in DataLoader(test_dataset, batch_size=32):
        X = X.to(device)
        preds.append(model(X).cpu().numpy())
        actual.append(y.numpy())

preds = np.vstack(preds)
actual = np.vstack(actual)

rmse = np.sqrt(mean_squared_error(actual.flatten(), preds.flatten()))
mase_score = mase(actual.flatten(), preds.flatten(), train_df["target"].values)

print("\nDeep Model RMSE:", rmse)
print("Deep Model MASE:", mase_score)

# ============================================================
# ARIMA BASELINE
# ============================================================

arima_model = ARIMA(train_df["target"], order=(5,1,0))
arima_fit = arima_model.fit()
forecast = arima_fit.forecast(steps=len(test_df))

baseline_rmse = np.sqrt(mean_squared_error(test_df["target"], forecast))
print("ARIMA RMSE:", baseline_rmse)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(actual.flatten()[:200], label="Actual")
plt.plot(preds.flatten()[:200], label="Predicted")
plt.legend()
plt.title("Forecast vs Actual")
plt.show()

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\nRunning SHAP explainability...")

background = train_dataset.X[:100].reshape(100, -1)
explainer = shap.DeepExplainer(model, background.to(device))

sample = test_dataset.X[:50].reshape(50, -1).to(device)
shap_values = explainer.shap_values(sample)

mean_shap = np.mean(np.abs(shap_values[0]), axis=0)
feature_importance = pd.Series(mean_shap).sort_values(ascending=False)

print("\nTop 5 Feature Importances:")
for i in range(5):
    print(f"{i+1}. Feature {feature_importance.index[i]} — {feature_importance.iloc[i]:.4f}")

shap.summary_plot(shap_values[0], sample.cpu().numpy())
