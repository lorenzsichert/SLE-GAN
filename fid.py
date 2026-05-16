import os
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm

from models import Generator


class DatasetTransform(Dataset):
    def __init__(self, transform, csv_file, img_dir):
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_dir, row["filename"])).convert("RGB")
        label = self.df.iloc[index, 1:].to_numpy(dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_activations(dataloader, model, device, num_images=None):
    model.eval()
    activations = []
    count = 0
    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            if batch.size(1) == 1:
                batch = batch.repeat(1, 3, 1, 1)
            feat = model(batch)
            feat = feat.cpu().numpy()
            activations.append(feat)
            count += batch.size(0)
            if num_images is not None and count >= num_images:
                break
    return np.concatenate(activations, axis=0)[:num_images]


def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/G-512-Epoch-675.pth", help="Generator checkpoint path")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_real", type=int, default=None, help="Number of real images to use")
    parser.add_argument("--num_fake", type=int, default=1000, help="Number of fake images to generate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--csv", type=str, default="../datasets/Psychart/sorted_images.csv")
    parser.add_argument("--img_dir", type=str, default="../datasets")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--skip_layer_g", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    generator = Generator(
        nz=args.latent_dim, ngf=args.features, img_size=args.img_size,
        nc=3, num_classes=args.num_classes, skip_layer=args.skip_layer_g
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    generator.load_state_dict(state)
    generator.eval()
    print(f"Loaded generator from {args.ckpt}")

    # Load real dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = DatasetTransform(transform, args.csv, args.img_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False,
    )

    # InceptionV3 for feature extraction (pool3)
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception.fc = nn.Identity()
    inception.eval()

    # Real activations
    print("Computing real image activations...")
    real_acts = get_activations(dataloader, inception, device, num_images=args.num_real)
    mu_real = real_acts.mean(axis=0)
    sigma_real = np.cov(real_acts, rowvar=False)
    print(f"Real images: {real_acts.shape[0]}")

    # Generate fake images
    print(f"Generating {args.num_fake} fake images...")
    fake_acts_list = []
    n_batches = (args.num_fake + args.batch_size - 1) // args.batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            bs = min(args.batch_size, args.num_fake - i * args.batch_size)
            noise = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            labels = torch.zeros(bs, args.num_classes, 1, 1, device=device)
            fake = generator(noise, labels)
            if fake.size(1) == 1:
                fake = fake.repeat(1, 3, 1, 1)
            feat = inception(fake)
            fake_acts_list.append(feat.cpu().numpy())
    fake_acts = np.concatenate(fake_acts_list, axis=0)[:args.num_fake]
    mu_fake = fake_acts.mean(axis=0)
    sigma_fake = np.cov(fake_acts, rowvar=False)
    print(f"Fake images: {fake_acts.shape[0]}")

    # Compute FID
    fid = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
