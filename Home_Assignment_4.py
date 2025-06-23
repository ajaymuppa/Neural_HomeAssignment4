#3. Programming Task (Basic GAN Implementation)


#Install and Import Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

#Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
])

train_data = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

#Define Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

#Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

fixed_noise = torch.randn(64, 100).to(device)

#Training Loop
epochs = 100
losses_G = []
losses_D = []

os.makedirs("images", exist_ok=True)

for epoch in range(epochs + 1):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        z = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(z)

        real_loss = criterion(D(real_imgs), real)
        fake_loss = criterion(D(fake_imgs.detach()), fake)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        z = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    losses_G.append(g_loss.item())
    losses_D.append(d_loss.item())

    # Save image samples
    if epoch in [0, 50, 100]:
        with torch.no_grad():
            generated_imgs = G(fixed_noise).cpu()
        grid = np.transpose(torchvision.utils.make_grid(generated_imgs, nrow=8, normalize=True), (1, 2, 0))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"images/sample_epoch_{epoch}.png")
        plt.close()

    print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

#Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(losses_D, label="Discriminator Loss")
plt.plot(losses_G, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator and Discriminator Loss")
plt.legend()
plt.grid(True)
plt.savefig("images/loss_plot.png")
plt.show()



#4. Programming Task (Data Poisoning Simulation)

#Build a Basic Sentiment Classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample dataset
texts = [
    "I loved the movie!", "What a terrible film", "It was fantastic",
    "Worst acting ever", "Absolutely brilliant", "Not worth watching",
    "UC Berkeley is excellent", "UC Berkeley is awful", "UC Berkeley rocks", "Hate UC Berkeley"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(X, y, texts, test_size=0.3, random_state=42)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate before poisoning
y_pred = model.predict(X_test)
acc_before = accuracy_score(y_test, y_pred)
cm_before = confusion_matrix(y_test, y_pred)

#Poison the Training Data
poisoned_texts_train = texts_train.copy()
poisoned_labels = y_train.copy()

# Flip labels where "UC Berkeley" is mentioned
for i, text in enumerate(poisoned_texts_train):
    if "UC Berkeley" in text:
        poisoned_labels[i] = 1 - poisoned_labels[i]  # flip 0<->1

# Retrain with poisoned data
X_poisoned = vectorizer.transform(poisoned_texts_train)
model_poisoned = LogisticRegression()
model_poisoned.fit(X_poisoned, poisoned_labels)

# Evaluate after poisoning
X_test_vector = vectorizer.transform(texts_test)
y_pred_poisoned = model_poisoned.predict(X_test_vector)
acc_after = accuracy_score(y_test, y_pred_poisoned)
cm_after = confusion_matrix(y_test, y_pred_poisoned)


#Visualize Results
# Accuracy comparison
plt.figure(figsize=(6, 4))
plt.bar(["Before Poisoning", "After Poisoning"], [acc_before, acc_after], color=["green", "red"])
plt.ylabel("Accuracy")
plt.title("Accuracy Before vs After Poisoning")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

disp1 = ConfusionMatrixDisplay(cm_before, display_labels=["Negative", "Positive"])
disp1.plot(ax=axs[0], cmap="Blues", colorbar=False)
axs[0].set_title("Before Poisoning")

disp2 = ConfusionMatrixDisplay(cm_after, display_labels=["Negative", "Positive"])
disp2.plot(ax=axs[1], cmap="Oranges", colorbar=False)
axs[1].set_title("After Poisoning")

plt.tight_layout()
plt.show()
