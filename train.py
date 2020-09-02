# Standard
import os

# Installed
import hydra


# PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Own
from utils import *

class DCGAN():
	def __init__(self, cfg):
		self.cfg = cfg.parameters

	def build(self):
		# DCGAN model
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.netD = Discriminator(self.cfg.channel_img, 
								  self.cfg.features).to(self.device)
		self.netG = Generator(self.cfg.vector_dim, 
							  self.cfg.channel_img, 
							  self.cfg.features).to(self.device)

		# Optimizer
		self.optimizerD = optim.Adam(self.netD.parameters(), 
									 lr=self.cfg.lr_init, 
									 betas=(0.5, 0.999),
									 )
		self.optimizerG = optim.Adam(self.netG.parameters(), 
									 lr=self.cfg.lr_init, 
									 betas=(0.5, 0.999),
									 )

		# Loss
		self.criterion = nn.BCELoss()

		# Load mnist
		self._load_data()

		# Fixed noise vector for evaluation
		self.fixed_noise = torch.randn(self.cfg.batch_size, self.cfg.vector_dim, 1, 1).to(self.device)

		self.writer_real = SummaryWriter("./GAN_MNIST/test_real")
		self.writer_fake = SummaryWriter("./GAN_MNIST/test_fake")

	def train(self):
		for epoch in range(self.cfg.num_epochs):
			self.netG.train()
			self.netD.train()
			for idx, (imgs, _) in enumerate(self.dataloader):
				imgs = imgs.to(self.device)

				### Train Discriminator: max[log(D(x)) + log(1-D(G(z)))]
				self.netD.zero_grad()
				labels = (torch.ones(self.cfg.batch_size)*0.9).to(self.device)
				output = self.netD(imgs).reshape(-1)
				lossD_real = self.criterion(output, labels)
				D_x = output.mean().item() # mean confidence level of the discriminator

				noise = torch.randn(self.cfg.batch_size, self.cfg.vector_dim, 1, 1).to(self.device)
				fake = self.netG(noise)
				labels = (torch.ones(self.cfg.batch_size)*0.1).to(self.device)
				output = self.netD(fake.detach()) # no BP to Generator
				lossD_fake = self.criterion(output, labels)

				lossD = lossD_fake + lossD_real
				lossD.backward()
				self.optimizerD.step()

				### Train Generator: max log(D(G(z)))
				self.netG.zero_grad()
				labels = torch.ones(self.cfg.batch_size).to(self.device)
				output = self.netD(fake).reshape(-1)
				lossG = self.criterion(output, labels)
				lossG.backward()
				self.optimizerG.step()

				### Summary Writer + Evaluation
				if idx % self.cfg.verbose_step == 0:
					print("[{}/{}] [{}/{}] LD: {:.4f} LG: {:.4f} Conf: {:.4f}".format(
														epoch, 
														self.cfg.num_epochs,
														idx, 
														len(self.dataloader),
														lossD,
														lossG,
														D_x)
						)

					self.evaluate(imgs)

	@torch.no_grad()
	def evaluate(self, real_imgs):
		self.netG.eval()
		self.netD.eval()

		fake = self.netG(self.fixed_noise)
		img_grid_real = torchvision.utils.make_grid(real_imgs[:32], normalize=True)
		img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
		self.writer_fake.add_image("MNIST Fake images", img_grid_fake)
		self.writer_real.add_image("MNIST Real images", img_grid_real)

		self.netG.train()
		self.netD.train()

	def _load_data(self):
		augmentation = transforms.Compose([
			transforms.Resize(self.cfg.image_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, ), (0.5, )),
			])

		dataset = datasets.MNIST(
						root='./dataset', 
						train=True, 
						transform=augmentation, 
						download=True
						)

		self.dataloader = DataLoader(
				dataset, 
				batch_size=self.cfg.batch_size, 
				shuffle=True
				)
@hydra.main(config_path="./default.yaml")
def main(cfg):

	dcgan = DCGAN(cfg)
	dcgan.build()
	dcgan.train()

if __name__ == "__main__":
	main()
