#!/usr/bin/env python
# coding: utf-8

# In[10]:


import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import get_model, to_device, prepare_dataloader
import matplotlib.pyplot as plt
import matplotlib as mpl
from configs import get_config
from data_loader import KittiLoader
from torch.utils.data import DataLoader, ConcatDataset
from transforms import image_transforms


# In[11]:


def SSIM(self, x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)


# In[12]:


def gradient_x(self, img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx


# In[13]:


def gradient_y(self, img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy


# In[14]:


def disp_smoothness(self, disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

    return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]


# In[15]:


def apply_disparity(self, img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')


# In[16]:


if __name__== '__main__':
    params = {}
    params = get_config(params)
    model = get_model(params["model"]) 
    print(model)


# In[17]:


data_transform = image_transforms(mode=params["mode"])
print(params["input_height"])
print(params["input_width"])
datasets = KittiLoader(params["data_dir"], params["mode"], transform=data_transform)
datasets_no_trans = KittiLoader(params["data_dir"], params["mode"])
train_loader = DataLoader(datasets, batch_size=params["batch_size"],
                shuffle=True, num_workers=params["num_workers"],pin_memory=True)
sample = datasets_no_trans.__getitem__(0)
fig = plt.figure(figsize=(50, 50))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("left_image")
ax1.imshow(sample["left_image"])
ax1.axis("off")
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("right_image")
ax2.imshow(sample["right_image"])
ax2.axis("off")
plt.show()


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])


# In[ ]:


losses = []
best_loss = float('Inf')
for epoch in range(params["epochs"]):
    c_time = time.time()
    running_loss = 0.0
    model.train()
    for (i, data) in enumerate(train_loader):
        # Load data
        print(data)
        """
        data = to_device(data, params["device"])
        left = data['left_image']
        right = data['right_image']

        # One optimization iteration
        optimizer.zero_grad()
        disps = model(left)
        loss = loss_function(disps, [left, right])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        running_loss += loss.item()
        """

    # Estimate loss per image
    running_loss /= params["n_img"] / params["batch_size"]
    print ('Epoch:',epoch + 1, 'train_loss:', running_loss, 'time:', round(time.time() - c_time, 3),'s',)
    torch.save(model.state_dict(), params["model_path"][:-4] + '_last.pth')
print ('Finished Training. Best loss:', best_loss)
torch.save(model.state_dict(), params["model_path"][:-4] + '_last.pth')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




