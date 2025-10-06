# gan_mnist.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import os

# =========================
# 1. Configuration
# =========================
latent_dim = 100  # size of the random noise vector
img_size = 28
batch_size = 128
epochs = 50
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("generated_images", exist_ok=True)

# =========================
# 2. Data Loader (MNIST)
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize between -1 and 1
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =========================
# 3. Generator Network
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()  # output between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, img_size, img_size)
        return img


# =========================
# 4. Discriminator Network
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()  # probability (real/fake)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# =========================
# 5. Initialize models & optimizer
# =========================
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

# =========================
# 6. Training Loop
# =========================
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Ground truths
        real = torch.ones((imgs.size(0), 1), device=device)
        fake = torch.zeros((imgs.size(0), 1), device=device)

        # Real images
        real_imgs = imgs.to(device)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}")

    # Save sample image every 10 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            z = torch.randn(25, latent_dim, device=device)
            gen_imgs = generator(z).detach().cpu()
            grid = gen_imgs[:25]
            grid = (grid + 1) / 2  # de-normalize to 0–1
            fig, axs = plt.subplots(5, 5, figsize=(5, 5))
            for ax, img in zip(axs.flatten(), grid):
                ax.imshow(img.squeeze(), cmap="gray")
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"generated_images/epoch_{epoch+1}.png")
            plt.close()
