from torch import mean, optim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
import copy

from diffaug import DiffAugment

from models import Generator, interpolate
from models import Discriminator


# Use as many threads as possible
torch.set_num_threads(20)
torch.set_num_interop_threads(20)

n_epochs = 2000
b1 = 0.5
b2 = 0.99
lr_g = 0.0002
lr_d = 0.0002
latent_dim = 256
features = 16
img_size = 256
layer = 256
channels = 3
batch_size = 16
discriminator_batch_size = batch_size
sample_interval = 16

alpha_end = 200.0
alpha_incease = 0.0002
alpha_dropdown = 1.0
counting_alpha = 1.0

load_ckpt = True


# --- Dataset Loading ---
link = "../progressive-gan/iamkaikaisubset/"
split = "train"
image_tag = "image"


class DatasetTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        item = self.dataset[index][image_tag]
        if self.transform:
            item = self.transform(item)
        return item

def convert_to_rgb(x):
    return x.convert("RGB")


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset = torchvision.datasets.ImageFolder(link, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=discriminator_batch_size,
    num_workers=10,
    shuffle=True,
    pin_memory=True,
)



# --- Cuda Init ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")



generator = Generator(nz=latent_dim, ngf=features, img_size=img_size, nc=channels, layer=layer)
discriminator = Discriminator(ndf=features, nc=channels, img_size=img_size, layer=layer)

if load_ckpt:
    try:
        generator.load_state_dict(torch.load(f"ckpt2/G-{layer}.pth"))
        discriminator.load_state_dict(torch.load(f"ckpt2/D-{layer}.pth"))
    except:
        print("Models could not be loaded!")


generator.to(device)
discriminator.to(device)

generator_ema = copy.deepcopy(generator)


optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))


alpha = 1.0

fixed_noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)



for ep in range(n_epochs):
    print(f"Epoch {ep}:")


    i = 0
    for batch in dataloader:

        counting_alpha += alpha_incease
        if (counting_alpha >= alpha_end and layer < img_size):
            counting_alpha = 0.0
            alpha_incease *= alpha_dropdown
            torch.save(generator.state_dict(), f"G-{layer}.pth")
            torch.save(discriminator.state_dict(), f"D-{layer}.pth")
            print("layer")
            layer *= 2
            generator.add_layer()
            discriminator.add_layer()

            generator.to(device)
            discriminator.to(device)

            optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
            optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))
        alpha = min(max(0.0,counting_alpha),1.0)
        i += 1


        # Train Discriminator on Real Images
        discriminator.zero_grad()

        real = batch[0].to(device)
        real = interpolate(real, (layer, layer))
        real = DiffAugment(real, policy="color,translation")

        output_real = discriminator(real, alpha)

        loss_real = mean(nn.functional.relu(1 - output_real))
        loss_real.backward()

        
        # Train Discriminator on Fake Images
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake = generator(noise, alpha)
        fake = DiffAugment(fake, policy="color,translation")

        output_fake = discriminator(fake, alpha) 

        loss_fake = mean(nn.functional.relu(1 + output_fake))
        loss_fake.backward()
        optimizerD.step()
 

        # Train Generator with Discriminator
        generator.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        output = generator(noise, alpha)
        output = DiffAugment(output, policy="color,translation")
        output_fake = discriminator(output, alpha) 
        loss_generated = -mean(output_fake)
        loss_generated.backward()
        optimizerG.step()

        with torch.no_grad():
            for p_ema, p in zip(generator_ema.parameters(), generator.parameters()):
                ema_alpha = 0.999
                p_ema.data.mul_(ema_alpha).add_(p.data, alpha=1 - ema_alpha)



        if i % (sample_interval/4) == 0:
            print(f"Ep: {ep}, i: {i}/{len(dataloader)}, alpha: {counting_alpha:.3f}, D(r): {mean(output_real):.3f}, D(f): {mean(output_fake):.3f}, D Loss: {(loss_real + loss_fake)/2:.3f}, G Loss:  {loss_generated:.3f}")
        if i % sample_interval == 0:
            save_image(output, f"images/image-{ep}.png", normalize=True)
            with torch.no_grad():
                fixed = generator_ema(fixed_noise, alpha)
                save_image(fixed, f"images/image-f-{ep}.png", normalize=True)
        if i % (sample_interval * 4) == 0:
            torch.save(generator.state_dict(), f"ckpt2/G-{layer}.pth")
            torch.save(generator_ema.state_dict(), f"ckpt2/GE-{layer}.pth")
            torch.save(discriminator.state_dict(), f"ckpt2/D-{layer}.pth")
