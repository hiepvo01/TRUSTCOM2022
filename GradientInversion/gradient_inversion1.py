import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2
import numpy as np
import math
from scipy.signal import convolve2d

from aijack.attack import GradientInversion_Attack

class LeNet(nn.Module):
    def __init__(self, channel=1, hideen=588, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.BatchNorm2d(12),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            nn.BatchNorm2d(12),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in torch.load('../data/MNIST1.pth')]

generated_images = []
generated_labels = []

def estimate_noise(I):
  H, W = I.shape
  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return round(sigma, 2)

for i in range(len(train_loader)):
    client_img = []
    client_label = []
    dataiter = iter(train_loader[i])
    for batch_idx in range(len(train_loader[i])):
        print("Client "+ str(i) +" | batch "+str(batch_idx) + " | total imgs " + str(len(client_img)))
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        criterion = nn.CrossEntropyLoss()
        net = LeNet(channel=1, num_classes=10)
        net.to(device)
        pred = net(images[:batch_size])
        loss = criterion(pred, labels[:batch_size])
        received_gradients = torch.autograd.grad(loss, net.parameters())
        received_gradients = [cg.detach() for cg in received_gradients]

        gradinversion = GradientInversion_Attack(net, (1, 28, 28), num_iteration=1400,
                                            lr=1e2, log_interval=0,
                                            optimizer_class=torch.optim.SGD,
                                            distancename="l2", optimize_label=False,
                                            bn_reg_layers=[net.body[1], net.body[4], net.body[7], net.body[10]],
                                            group_num = 3,
                                            tv_reg_coef=0.00, l2_reg_coef=0.0001,
                                            bn_reg_coef=0.001, gc_reg_coef=0.001, device=device)

        result = gradinversion.group_attack(received_gradients, batch_size=batch_size)

        for bid in range(batch_size):
            test_img = torch.from_numpy(((sum(result[0]) / len(result[0])).cpu().detach().numpy()[bid]))
            test_img = cv2.medianBlur(test_img, 3)
            img1 = test_img.swapaxes(0,1)
            img1 = img1.swapaxes(1,2)
            if estimate_noise(np.array(cv2.medianBlur(img1,3))) < 0.5:
                client_img.append(img1)
                label = result[1][0][bid].item()    
                client_label.append(label)  
            
    generated_images.append(client_img)
    generated_labels.append(client_label)
torch.save(generated_images, 'generated_images3.pth')
torch.save(generated_labels, 'generated_labels3.pth')