import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = sns.load_dataset("penguins")
df = df.dropna()
print(df.size)

sns.pairplot(df, hue = "species")
plt.show()

X = pd.get_dummies(df.drop("species", axis=1))
encoder = LabelEncoder()
y = encoder.fit_transform(df["species"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


class PenguinNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

model = PenguinNet(input_dim=X_train.shape[1], num_classes=len(np.unique(y)))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 300
for epoch in range(epochs):
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss={loss.item():.4f}")


model.eval()

with torch.no_grad():
    test_preds = model(X_test).argmax(axis = 1)
    test_acc = (test_preds == y_test).float().mean()

    print(f"Test accuracy: {test_acc:.3f}")

    preds = model(X_test).argmax(1).numpy()

cm = confusion_matrix(y_test.numpy(), preds)
print(cm)

disp = ConfusionMatrixDisplay(cm, display_labels=encoder.classes_)
disp.plot(cmap="Blues")
plt.show()
