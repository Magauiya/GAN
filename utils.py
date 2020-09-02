import torch
import torch.nn as nn


class Discriminator(nn.Module):
	def __init__(self, channels, features):
		super().__init__()
		self.net = nn.Sequential(
			# N x channels x 64 x 64 
			nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*2),
			nn.LeakyReLU(0.2),
			nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*4),
			nn.LeakyReLU(0.2),
			nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*8),
			nn.LeakyReLU(0.2),
			nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
			# N x 1 x 1 x 1
			nn.Sigmoid()
			)

	def forward(self, x):
		return self.net(x)


class Generator(nn.Module):
	def __init__(self, channels_in, channels_out, features):
		super().__init__()
		self.net = nn.Sequential(
			# N x channels_in x 1 x 1
			nn.ConvTranspose2d(channels_in, features*16, kernel_size=4, stride=2, padding=0),
			nn.BatchNorm2d(features*16),
			nn.ReLU(),
			# N x features*16 x 4 x 4
			nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*8),
			nn.ReLU(),
			# N x features*8 x 8 x 8
			nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*4),
			nn.ReLU(),
			# N x features*4 x 16 x 16
			nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features*2),
			nn.ReLU(),
			# N x features*2 x 32 x 32
			nn.ConvTranspose2d(features*2, channels_out, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
			)

	def forward(self, x):
		return self.net(x)