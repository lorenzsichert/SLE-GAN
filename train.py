from torch import mean, optim, randint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from torch.nn.functional import interpolate
import copy

import lpips



from diffaug import DiffAugment

from models import Generator
from models import Discriminator


# Use as many threads as possible
torch.set_num_threads(20)
torch.set_num_interop_threads(20)

b1 = 0.5
b2 = 0.99
lr_g = 0.0002
lr_d = 0.0002
ema_alpha = 0.999

latent_dim = 256
features = 16
img_size = 256
layer = 256
channels = 3

num_classes = 10

batch_size = 1
discriminator_batch_size = batch_size
sample_interval = 16
ckpt_interval = 1024


n_epochs = 2000
start_ep = 0
load_ckpt = False


# --- Dataset Loading ---
link = "../progressive-gan/album_covers_sorted/"
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

if torch.cuda.is_available():
    percept = lpips.LPIPS(net="vgg").cuda()
else:
    percept = lpips.LPIPS(net="vgg")



generator = Generator(nz=latent_dim, ngf=features, img_size=img_size, nc=channels, layer=layer, num_classes=num_classes)
discriminator = Discriminator(ndf=features, nc=channels, img_size=img_size, layer=layer, num_classes=num_classes)

if load_ckpt:
    try:
        generator.load_state_dict(torch.load(f"ckpt2/G-{layer}.pth"))
        discriminator.load_state_dict(torch.load(f"ckpt2/D-{layer}.pth"))
        print("Models loaded from file!")
    except:
        print("Models could not be loaded!")
else:
    print("Not loading models!")


generator.to(device)
discriminator.to(device)

generator_ema = copy.deepcopy(generator)


optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))


alpha = 1.0

fixed_noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)

iteration = 0


for ep in range(start_ep, n_epochs):
    print(f"Epoch {ep}:")
    print(dataset.classes)


    i = 0
    for batch, labels in dataloader:
        iteration += 1
        i += 1


        # Train Discriminator on Real Images
        discriminator.zero_grad()

        for i in labels:
            print(i)
            print(dataset.classes[i])
        real = batch.to(device)
        real = interpolate(real, (layer, layer))
        real_128 = interpolate(real, size=128)
        real = DiffAugment(real, policy="color,translation")
        real_128 = DiffAugment(real_128, policy="color,translation")

        part = randint(0,8,(1,2))[0].to(device)

        output_real, [rec_small, rec_big, rec_part] = discriminator(real, real_128, label="real", part=part)




        real_part = interpolate(real, size=256)
        real_part = real_part[:,:,part[0]*16:part[0]*16+128,part[1]*16:part[1]*16+128]

        loss_real = mean(nn.functional.relu(1 - output_real)) +\
            percept(rec_small, real_128).sum() +\
            percept(rec_big, real_128).sum() +\
            percept(rec_part, real_part).sum()
        loss_real.backward()

        
        # Train Discriminator on Fake Images
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake, fake_128 = generator(noise)
        fake = DiffAugment(fake, policy="color,translation")
        fake_128 = DiffAugment(fake_128, policy="color,translation")

        output_fake = discriminator(fake, fake_128) 

        loss_fake = mean(nn.functional.relu(1 + output_fake))
        loss_fake.backward()
        optimizerD.step()
 

        # Train Generator with Discriminator
        generator.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        output, output_128 = generator(noise)
        output = DiffAugment(output, policy="color,translation")
        output_128 = DiffAugment(output_128, policy="color,translation")
        output_fake = discriminator(output, output_128) 
        loss_generated = -mean(output_fake)
        loss_generated.backward()
        optimizerG.step()

        with torch.no_grad():
            for p_ema, p in zip(generator_ema.parameters(), generator.parameters()):
                p_ema.data.mul_(ema_alpha).add_(p.data, alpha=1 - ema_alpha)
            for b_ema, b in zip(generator_ema.buffers(), generator.buffers()):
                b_ema.data.copy_(b.data)




        if i % 4 == 0:
            print(f"Ep: {ep}, i: {i}/{len(dataloader)}, iteration: {iteration}, D(r): {mean(output_real):.3f}, D(f): {mean(output_fake):.3f}, D Loss: {(loss_real + loss_fake)/2:.3f}, G Loss:  {loss_generated:.3f}")
        if iteration % sample_interval == 0:
            save_image(output, f"images/image-{ep}.png", normalize=True)
            save_image(output_128, f"images/image-128-{ep}.png", normalize=True)
            out = torch.cat([real_128, rec_part, rec_big, rec_small])
            save_image(out, f"images/image-rec-{ep}.png", nrow=batch_size, normalize=True)
            with torch.no_grad():
                fixed, fixed_128 = generator_ema(fixed_noise)
                save_image(fixed, f"images/image-f-{ep}.png", normalize=True)
                save_image(fixed_128, f"images/image-128-f-{ep}.png", normalize=True)
        if iteration % ckpt_interval == 0:
            print("save dict")
            torch.save(generator.state_dict(), f"ckpt/G-{layer}-Epoch-{ep}.pth")
            torch.save(generator_ema.state_dict(), f"ckpt/GE-{layer}-Epoch-{ep}.pth")
            torch.save(discriminator.state_dict(), f"ckpt/D-{layer}-Epoch-{ep}.pth")
