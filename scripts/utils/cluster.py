import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import shutil

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

class ImageDataset(Dataset):
    def __init__(self, image_dir, target_size=(32, 32)):
        self.image_dir = image_dir
        self.target_size = target_size
        self.file_names = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
        self.images = []
        for file_name in self.file_names:
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = np.zeros_like(image, dtype=np.uint8)
            cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
            image = contour_image

            h, w = image.shape
            if h > target_size[0] or w > target_size[1]:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                h, w = target_size
            padded_image = np.zeros(target_size, dtype=np.uint8)
            start_y = (target_size[0] - h) // 2
            start_x = (target_size[1] - w) // 2
            padded_image[start_y:start_y + h, start_x:start_x + w] = image
            self.images.append(padded_image)

        self.images = np.array(self.images, dtype=np.float32) / 255.0  
        self.images = np.expand_dims(self.images, axis=1)  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

def train_vae(vae, dataloader, optimizer, epochs, device):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(images)

            recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def cluster_and_save(vae, dataset, output_dir, clu_num, device):
    vae.eval()
    images = torch.tensor(dataset.images).to(device)
    with torch.no_grad():
        mu, _ = vae.encode(images)
        z = mu.cpu().numpy()

    original_sizes = np.array([cv2.imread(os.path.join(dataset.image_dir, f), cv2.IMREAD_GRAYSCALE).shape for f in dataset.file_names])
    original_sizes = original_sizes.astype(np.float32)

    features = np.hstack((z, original_sizes))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=clu_num, random_state=0)
    labels = kmeans.fit_predict(features)

    cluster_dirs = [os.path.join(output_dir, f"cluster_{i}") for i in range(clu_num)]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cluster_dir in cluster_dirs:
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

    for i, (label, image) in enumerate(zip(labels, dataset.images)):
        image = (image * 255).astype(np.uint8).squeeze(0)  
        output_path = os.path.join(cluster_dirs[label], f"image_{i}.bmp")
        cv2.imwrite(output_path, image)
    print(f"Images have been saved to {output_dir} by cluster.")

if __name__ == "__main__":
    image_dir = "./taxi-mutation"
    output_dir = "./clustered_images"
    latent_dim = 16
    clu_num = 2
    batch_size = 16
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    train_vae(vae, dataloader, optimizer, epochs, device)

    cluster_and_save(vae, dataset, output_dir, clu_num, device)
