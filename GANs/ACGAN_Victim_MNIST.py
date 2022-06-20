# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnist
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from matplotlib import pyplot
import os
import random
import torch
import cv2
import torch.nn as nn
import math
from scipy.signal import convolve2d


def estimate_noise(I):
  H, W = I.shape
  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return round(sigma, 2)

images = torch.load('../GradientInversion/generated_images1.pth')
images1 = torch.load('../GradientInversion/generated_images2.pth')

for i in range(10):
  images[i] += images1[i]

for i in range(10):
  imgs = images[i]
  imgs = [img for img in imgs if estimate_noise(np.array(cv2.medianBlur(img.numpy(),3))) < 0.5]
  images[i] = imgs

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

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10, n_victims=10):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 14x14
	fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# downsample to 7x7
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# real/fake output
	out1 = Dense(1, activation='sigmoid')(fe)
	# class label output
	out2 = Dense(n_classes, activation='softmax')(fe)
	out3 = Dense(n_victims, activation='softmax')(fe)
	# define model
	model = Model(in_image, [out1, out2, out3])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=10, n_victims=10):
	# weight initialization
	init = RandomNormal(stddev=0.02)
 
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)    
    # linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes, kernel_initializer=init)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
 
 # victim input
	in_victim = Input(shape=(1,))
	# embedding for categorical input
	vi = Embedding(n_victims, 50)(in_victim)    
    # linear multiplication
	n_nodes = 7 * 7
	vi = Dense(n_nodes, kernel_initializer=init)(vi)
	# reshape to additional channel
	vi = Reshape((7, 7, 1))(vi)
 
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 384 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((7, 7, 384))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li, vi])
	# upsample to 14x14
	gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model([in_lat, in_label, in_victim], out_layer)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect the outputs of the generator to the inputs of the discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and label and outputting real/fake and label outputs
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# load images
def load_real_samples():
    # load dataset
    train_x = []
    train_y = []
    train_victim = []
    for client in range(10):
        for idx in range(len(images[client])):
            img = cv2.medianBlur(images[client][idx].numpy(),3)
            test1 = images[client][idx].numpy()
            # test1 = np.expand_dims(img, axis=0)
            # test1 = np.transpose(test1, (1, 2, 0))
            train_x.append(test1)
            
            test = np.expand_dims(img, axis=0)
            test = np.expand_dims(test, axis=0)
            model = torch.load("../pretrained/test_torch.pt")
            tc_img = torch.tensor(test).cuda()
            output = model(tc_img)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = pred.cpu().detach().numpy()[0][0]
            train_y.append(pred)
        train_victim += [client]*len(images[client])

    return [np.array(train_x), np.array(train_y), np.array(train_victim)]

# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels, victims = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    
    X, labels, victims = images[ix], labels[ix], victims[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels, victims], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10, n_victims=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
 # generate victims
	victims = randint(0, n_victims, n_samples)
	return [z_input, labels, victims]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input, victims_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input, victims_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input, victims_input], y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, n_samples=100):
	# prepare fake examples
	[X, _, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'results/ACGAN/generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'results/ACGAN/g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
  # save the discriminator model
	filename3 = 'results/ACGAN/d_model_%04d.h5' % (step+1)
	d_model.save(filename3)
	print('>Saved: %s and %s' % (filename1, filename2))

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=10):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# get randomly selected 'real' samples
		#####################################################################################################################
		#####################################################################################################################
		[X_real, labels_real, victims_real], y_real = generate_real_samples(dataset, half_batch)
		#####################################################################################################################
		#####################################################################################################################
		# update discriminator model weights
		_,d_r1,d_r2,d_r3 = d_model.train_on_batch(X_real, [y_real, labels_real, victims_real])
		# generate 'fake' examples
		[X_fake, labels_fake, victims_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		_,d_f,d_f2,d_f3 = d_model.train_on_batch(X_fake, [y_fake, labels_fake, victims_fake])
		# prepare points in latent space as input for the generator
		[z_input, z_labels, z_victims] = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		_,g_1,g_2,g_3 = gan_model.train_on_batch([z_input, z_labels, z_victims], [y_gan, z_labels, z_victims])
		# summarize loss on this batch
		print('>%d, dr[%.3f,%.3f,%.3f], df[%.3f,%.3f,%.3f], g[%.3f,%.3f,%.3f]' % (i+1, d_r1,d_r2,d_r3, d_f,d_f2,d_f3, g_1,g_2,g_3))
		# evaluate the model performance every 'epoch'
		if (i+1) % 1000 == 0:
			summarize_performance(i, g_model, d_model, latent_dim)
   
# make folder for results
os.makedirs('results/ACGAN', exist_ok=True)
# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)
