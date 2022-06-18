# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.utils import save_image

# from aijack.attack import GradientInversion_Attack

# class LeNet(nn.Module):
#     def __init__(self, channel=1, hideen=588, num_classes=10):
#         super(LeNet, self).__init__()
#         act = nn.Sigmoid
#         self.body = nn.Sequential(
#             nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
#             nn.BatchNorm2d(12),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
#             nn.BatchNorm2d(12),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
#             nn.BatchNorm2d(12),
#             act(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hideen, num_classes)
#         )

#     def forward(self, x):
#         out = self.body(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# batch_size = 4
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in torch.load('../data/cifar10.pth')]

# generated_images = []
# generated_labels = []

# for i in range(len(train_loader)):
#     client_img = []
#     client_label = []
#     dataiter = iter(train_loader[i])
#     for batch_idx in range(len(train_loader[i])):
#         print("Client "+ str(i) +" | batch "+str(batch_idx) + " | total imgs " + str(len(client_img)))
#         images, labels = dataiter.next()
#         images = images.cuda()
#         labels = labels.cuda()

#         criterion = nn.CrossEntropyLoss()
#         net = LeNet(channel=3, hideen=768, num_classes=10)
#         net.to(device)
#         pred = net(images[:batch_size])
#         loss = criterion(pred, labels[:batch_size])
#         received_gradients = torch.autograd.grad(loss, net.parameters())
#         received_gradients = [cg.detach() for cg in received_gradients]

#         gradinversion = GradientInversion_Attack(net, (3, 32, 32), num_iteration=1000,
#                                             lr=1e2, log_interval=0,
#                                             optimizer_class=torch.optim.SGD,
#                                             distancename="l2", optimize_label=False,
#                                             bn_reg_layers=[net.body[1], net.body[4], net.body[7]],
#                                             group_num = 3,
#                                             tv_reg_coef=0.00, l2_reg_coef=0.0001,
#                                             bn_reg_coef=0.001, gc_reg_coef=0.001, device=device)

#         result = gradinversion.group_attack(received_gradients, batch_size=batch_size)

#         for bid in range(batch_size):
#             test_img = torch.from_numpy(((sum(result[0]) / len(result[0])).cpu().detach().numpy()[bid]))
#             img1 = test_img.swapaxes(0,1)
#             img1 = img1.swapaxes(1,2)
#             client_img.append(img1)
#             label = result[1][0][bid].item()
#             client_label.append(label)
#     generated_images.append(client_img)
#     generated_labels.append(client_label)
# torch.save(generated_images, 'generated_images.pth')
# torch.save(generated_labels, 'generated_labels.pth')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from aijack.attack import GradientInversion_Attack

class LeNet(nn.Module):
    def __init__(self, channel=1, hideen=588, num_classes=10):
        super(LeNet, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=(3,3), stride=(2,2), padding=1),
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
            nn.Linear(16384, num_classes)
            # nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in torch.load('../data/cifar10.pth')]

generated_images = []
generated_labels = []

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
        net = LeNet(channel=3, hideen=768, num_classes=10)
        net.to(device)
        pred = net(images[:batch_size])
        loss = criterion(pred, labels[:batch_size])
        received_gradients = torch.autograd.grad(loss, net.parameters())
        received_gradients = [cg.detach() for cg in received_gradients]

        gradinversion = GradientInversion_Attack(net, (3, 32, 32), num_iteration=1000,
                                            lr=1e2, log_interval=0,
                                            optimizer_class=torch.optim.SGD,
                                            distancename="l2", optimize_label=False,
                                            bn_reg_layers=[net.body[4], net.body[8], net.body[12]],
                                            group_num = 3,
                                            tv_reg_coef=0.00, l2_reg_coef=0.0001,
                                            bn_reg_coef=0.001, gc_reg_coef=0.001, device=device)

        result = gradinversion.group_attack(received_gradients, batch_size=batch_size)
        for bid in range(batch_size):
            test_img = torch.from_numpy(((sum(result[0]) / len(result[0])).cpu().detach().numpy()[bid]))
            img1 = test_img.swapaxes(0,1)
            img1 = img1.swapaxes(1,2)
            
            client_img.append(img1)
            label = result[1][0][bid].item()
            client_label.append(label)

    generated_images.append(client_img)
    generated_labels.append(client_label)
torch.save(generated_images, 'generated_images2.pth')
torch.save(generated_labels, 'generated_labels2.pth')