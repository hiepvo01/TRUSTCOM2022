# example of a wgan for generating handwritten digits
from numpy import expand_dims, mean, ones, zeros
from numpy.random import randn, randint
import os

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import backend
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout, Concatenate, Embedding, Input

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model

# define the combined generator and critic model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


# # load images
# def load_real_samples():
# 	# load dataset
# 	(trainX, trainy), (_, _) = load_data()
# 	# expand to 3d, e.g. add channels
# 	X = expand_dims(trainX, axis=-1)
# 	# convert from ints to floats
# 	X = X.astype('float32')
# 	# scale from [0,255] to [-1,1]
# 	X = (X - 127.5) / 127.5
# 	return [X, trainy]

import numpy as np
import torch
import torch.nn as nn
import cv2
import math
from scipy.signal import convolve2d
from torchvision.transforms.functional import rotate

class VGG(nn.Module):
    def __init__(self, channels=1, hideen=12544, num_classes=10):
        super(VGG, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(12544, num_classes)
            # nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, n_samples=100):
	# prepare fake examples
	[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i], cmap='gray_r')
	# save plot to file
	filename1 = 'results/CGAN/generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'results/CGAN/g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
  # save the discriminator model
	filename3 = 'results/CGAN/d_model_%04d.h5' % (step+1)
	d_model.save(filename3)
	print('>Saved: %s and %s' % (filename1, filename2))
    

def estimate_noise(I):
  H, W = I.shape
  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return round(sigma, 2)

images = torch.load('../GradientInversion/generated_images3.pth')
images1 = torch.load('../GradientInversion/generated_images2.pth')

for i in range(10):
    for item in range(len(images[i])):
        images[i][item] = rotate(images[i][item], 0.2*i)
    for item in range(len(images1[i])):
        images1[i][item] = rotate(images1[i][item], 0.2*i)
    images[i] += images1[i]

for i in range(10):
  imgs = images[i]
  imgs = [np.expand_dims(np.array(cv2.medianBlur(img.numpy(),3)), axis=0) for img in imgs if estimate_noise(np.array(cv2.medianBlur(img.numpy(),3))) < 0.5]
  images[i] = imgs
  
train_loader = [torch.utils.data.DataLoader(x, batch_size=1, shuffle=True) for x in torch.load('../data/MNIST.pth')]
for i in range(len(train_loader)):
    dataiter = iter(train_loader[i])
    for batch_idx in range(len(train_loader[i])):
        image, label = dataiter.next()
        images[i].append(image[0].numpy())  

# load images
def load_real_samples(n_victims=10):
    # load dataset
    train_x = []
    train_y = []
    train_victim = []
    for client in range(10):
        for idx in range(len(images[client])):
            img = images[client][idx]
            train_x.append(np.transpose(img, (1,2,0)))
            
            model = torch.load("../pretrained/test_torch.pt")
            test1 = np.expand_dims(img, axis=0)
            tc_img = torch.tensor(test1).cuda()
            output = model(tc_img)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.cpu().detach().numpy()[0][0]
            train_y.append(pred)
        if n_victims > 1:
            train_victim += [client]*len(images[client])
        else:
            if client == 0:
                train_victim = [1]*len(images[client])
            else:
                train_victim += [0]*len(images[client])
    print(len(train_x))
    return [np.array(train_x), np.array(train_y)]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=25):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
			(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		print((i+1) % 10)
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, 100)
	# g_model.save('results/CGAN/cgan_generator.h5')

# make folder for results
os.makedirs('results/CGAN', exist_ok=True)
# size of the latent space
latent_dim = 100
# create the critic
critic = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples()
# train model
train(generator, critic, gan_model, dataset, latent_dim)