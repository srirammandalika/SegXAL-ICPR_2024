#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm


# In[2]:


get_ipython().run_line_magic('cd', '')


# In[3]:


data_dir = os.path.join("/Users/srirammandalika/Downloads/Cityscapes")
train_dir = os.path.join(data_dir, "/Users/srirammandalika/Downloads/Cityscapes/train") 
val_dir = os.path.join(data_dir, "/Users/srirammandalika/Downloads/Cityscapes/val")
train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)
print(len(train_fns), len(val_fns))


# In[4]:


sample_image_fp = os.path.join(train_dir, train_fns[0])
sample_image = Image.open(sample_image_fp).convert("RGB")


# In[5]:


"""
color_set = set()
for train_fn in tqdm(train_fns[:10]):
    train_fp = os.path.join(train_dir, train_fn)
    image = np.array(Image.open(train_fp))
    cityscape, label = split_image(sample_image)
    label = label.reshape(-1, 3)
    local_color_set = set([tuple(c) for c in list(label)])
    color_set.update(local_color_set)
color_array = np.array(list(color_set))
"""

num_items = 1200
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)
print(color_array[:5, :])


# In[6]:


num_classes = 10
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)


# In[7]:


class UNet(nn.Module):
    
    def __init__(self, num_classes):
        
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out
    
#     def entropy(self, predictions):
#         # calculate the entropy score of the predictions
#         # assuming the predictions are of shape (batch_size, num_classes, H, W)
#         entropy = -torch.sum(predictions * torch.log(predictions + 1e-7), dim=1)
#         return entropy.mean()


# In[8]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# In[9]:


class CityscapeDataset(Dataset):

    
    
    def __init__(self, image_dir, label_model):
        self.image_dir = image_dir
        self.image_fns = os.listdir(image_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape, label_class
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)


# In[10]:


import torch


model_path = "/Users/srirammandalika/Desktop/U-Net.pth"
device = torch.device("cpu")

model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path, map_location=device))


# In[11]:


test_batch_size = 16
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size)


# In[12]:


X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)


# In[13]:


inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])


# In[14]:


import torch
import torch.nn.functional as F

def calculate_entropy(outputs, targets):
    # Convert the outputs to probabilities by applying a softmax function
    outputs = torch.softmax(outputs, dim=1)
    
    # Extract the probabilities for the correct class from the output
    correct_class_probs = outputs.gather(1, targets.unsqueeze(1))
    
    # Calculate the entropy for each pixel
    entropy = -1 * correct_class_probs * torch.log(correct_class_probs + 1e-30)
    
    return entropy


# In[15]:


class EntropyLoss(nn.Module):
    def _init_(self):
        super(EntropyLoss, self).init()
    def forward(self, pred, target):
    # convert target to one-hot encoding
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
    # calculate the negative of element-wise product of predicted probability and log of predicted probability
        entropy = -1 * (target_onehot * torch.log(pred+1e-30)).sum(dim=1).mean()
        return entropy


# In[ ]:





# In[16]:


criterion = nn.CrossEntropyLoss()
entropy_loss_fn = EntropyLoss()


# In[17]:


device = torch.device("cpu")

model = UNet(num_classes).to(device)
model.load_state_dict(torch.load("/Users/srirammandalika/Desktop/U-Net.pth", map_location=device))
model.eval()


# In[18]:


output = model(X.cpu())
entropy = calculate_entropy(output,Y_pred.cpu())


# In[19]:


print(entropy[0].shape)
e = entropy.cpu().detach().numpy()
print(e.shape)
# print(e.shape)
for i in range(16):
    plt.imshow(e[i][0])
    plt.show()
# fig, axes = plt.subplots(test_batch_size, 1, figsize=(1*5, test_batch_size*5))
# for i in range(test_batch_size):
#     axes[i].imshow(e[0])


# In[20]:


entropy_grid = torchvision.utils.make_grid(entropy[0])
plt.imshow(entropy_grid.permute(1, 2, 0))
plt.show()


# In[21]:


e_grid = torchvision.utils.make_grid(entropy[0])
plt.imshow(e_grid[0])


# In[22]:


# print(entropy_grid.shape)
type(X[0])


# 

# In[28]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(test_batch_size, 4, figsize=(4*5, test_batch_size*5))

for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i]  # Directly use Y_pred as it's already a numpy array
    
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")
    axes[i, 3].imshow(e[i][0])
    axes[i, 3].set_title("Entropy")

    # Save the landscape image to a file
    plt.imsave(f'/Users/srirammandalika/Downloads/unlabeled_data/Loop1/predicted-label/{i}_label.png', label_class_predicted)

plt.tight_layout()
plt.show()


# In[29]:


fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")


# In[25]:


def entropy(images,net):
    X, Y = next(iter(data_loader))
    X, Y = X.to(device).float(), Y.to(device).long()
    pred = model_(X)
    pred = F.relu(pred)
    pred = pred.detach()
    pred = torch.max(pred[:,:,:],dim=1)
    pred = pred[0]
    
    entropy = -1 * torch.sum(pred * torch.log(pred+1e-30))
    entropy = entropy.detach().cpu().numpy()
    return entropy
entropy(val_fns,model_)


# In[26]:


def entropy(pred):
    
    pred = pred.detach()
    pred = torch.max(pred[:,:,:],dim=1)
    pred = pred[0]
    
    entropy = (-1) * torch.sum(pred * torch.log(pred+1e-30))
    entropy = entropy.detach().cpu().numpy()
    return entropy
entropy(Y_pred)


# In[27]:


label_class_predicted = []
for i in range(test_batch_size):
    label_class_predicted.append(Y_pred[i].cpu().detach().numpy())
most_conf = np.nanmax(label_class_predicted)
num_labels = len(label_class_predicted)
numerator = (num_labels*(1 - most_conf))
denominator = (num_labels-1)

least_conf = numerator/denominator
print(least_conf)


# In[29]:


print(Y_pred.shape)


# In[58]:


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from scipy.stats import entropy

def patchwise_entropy(image, patch_size):
    """
    Calculates entropy for each patch in the image.
    Args:
    - image (numpy array): The input image.
    - patch_size (int): The size of each square patch.
    
    Returns:
    - List of entropy values for each patch.
    """
    entropy_vals = []
    for x in range(0, image.shape[0] - patch_size + 1, patch_size):
        for y in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[x:x + patch_size, y:y + patch_size]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            entropy_vals.append(entropy(hist))
    return entropy_vals


fig, axes = plt.subplots(test_batch_size, 4, figsize=(4*5, test_batch_size*5))

for i in range(test_batch_size):
    # Assuming you have the images in the correct format
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i]

    # Calculate entropy and create an entropy map for the current segmented image
    entropy_vals = patchwise_entropy(label_class_predicted, 8)  # Pass 8 as patch_size
    entropy_map = np.zeros(label_class_predicted.shape)
    k = 0
    for x in range(0, label_class_predicted.shape[0] - 8 + 1, 8):
        for y in range(0, label_class_predicted.shape[1] - 8 + 1, 8):
            entropy_map[x:x + 8, y:y + 8] = entropy_vals[k]
            k += 1

    # Define high entropy threshold
    high_entropy_threshold = np.mean(entropy_vals) + np.std(entropy_vals)

    # Create a mask for high entropy patches
    high_entropy_mask = entropy_map > high_entropy_threshold

    # Replace high entropy patches in label_class_predicted with corresponding patches from label_class
    modified_segmented_image = np.copy(label_class_predicted)
    for x in range(0, label_class_predicted.shape[0] - 8 + 1, 8):
        for y in range(0, label_class_predicted.shape[1] - 8 + 1, 8):
            if high_entropy_mask[x:x + 8, y:y + 8].any():
                modified_segmented_image[x:x + 8, y:y + 8] = label_class[x:x + 8, y:y + 8]

    # Plotting
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")
    axes[i, 3].imshow(modified_segmented_image)
    axes[i, 3].set_title("Modified Segmented Image")

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Applying ViewAL approach for Viewpoint entropy & superpixel oriented uncertainty

# In[26]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class ViewAL(nn.Module):
    def __init__(self, num_classes, viewpoints):
        super(ViewAL, self).__init__()
        self.num_classes = num_classes
        self.viewpoints = viewpoints

    def forward(self, predictions, views):
        # Compute viewpoint entropy
        prob = F.softmax(predictions, dim=1)
        prob_view = prob.gather(2, views.view(-1, 1, 1, 1).expand(-1, self.num_classes, -1, -1))
        log_prob_view = torch.log(prob_view + 1e-10)
        entropy_view = -(prob_view * log_prob_view).mean(dim=1).mean(dim=1)
        
        # Compute superpixel-based uncertainty
        _, _, H, W = predictions.size()
        superpixels = self.compute_superpixels(predictions)  # (batch_size, num_superpixels, num_classes)
        prob_superpixel = F.softmax(superpixels, dim=2)
        entropy_superpixel = -(prob_superpixel * torch.log(prob_superpixel + 1e-10)).sum(dim=2)
        
        # Combine viewpoint entropy and superpixel-based uncertainty
        uncertainty = entropy_view + entropy_superpixel
        
        return uncertainty
    
    def compute_superpixels(self, predictions):
        pass
# Compute superpixels from predictions.


# # Entropy Score...

# In[31]:


pip install scikit-image


# In[31]:


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities + 1e-10)).sum()

# output
#prob = torch.tensor(Y_pred)
entropy_score = entropy(Y_pred[0])
print(entropy_score)


# In[45]:


from skimage.io import imread, imshow
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


# In[46]:


def select_uncertain_pixels(masked_image, model):
    # Get the predicted probability for each pixel
    predicted_probabilities = model(masked_image)
    # Compute the entropy for each pixel
    pixel_entropies = -np.sum(predicted_probabilities * np.log(predicted_probabilities + 1e-10), axis=-1)
    # Normalize the entropies to be between 0 and 1
    pixel_entropies = (pixel_entropies - np.min(pixel_entropies)) / (np.max(pixel_entropies) - np.min(pixel_entropies))
    # Select the top N uncertain pixels
    N = 1200
    uncertain_pixels = np.argpartition(pixel_entropies.ravel(), -N)[-N:]
    return uncertain_pixels


# In[47]:


import cv2

def overlay_heatmap(segmented_image, uncertain_pixels):
    # Create an empty heatmap image with the same shape as the segmented image
    heatmap = np.zeros_like(segmented_image)
    # Set the value of the uncertain pixels to 255
    heatmap[uncertain_pixels] = 255
    # Convert the heatmap to a color image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Overlay the heatmap on top of the segmented image
    heatmap = cv2.addWeighted(segmented_image, 0.5, heatmap, 0.5, 0)
    return heatmap


# In[48]:


def select_uncertain_pixels(input_tensor, model):
    # Convert the image to a tensor
#     input_tensor = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float()
    # Get the predicted probability for each pixel
    predicted_probabilities = model(input_tensor)
    # Compute the entropy for each pixel
    pixel_entropies = -torch.sum(predicted_probabilities * torch.log(predicted_probabilities + 1e-10), dim=-1)
    # Normalize the entropies to be between 0 and 1
    pixel_entropies = (pixel_entropies - torch.min(pixel_entropies)) / (torch.max(pixel_entropies) - torch.min(pixel_entropies))
    # Select the top N uncertain pixels
    N = 1000
    uncertain_pixels = torch.argsort(pixel_entropies.view(-1), descending=True)[:N]
    return uncertain_pixels


# In[49]:


select_uncertain_pixels(X,model_)


# In[50]:


def overlay_heatmap(segmented_image, entropies, cmap='jet'):
    # Normalize the entropies to be between 0 and 1
    entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
    # Convert the entropies to a 2D image
    entropies_image = entropies.reshape(segmented_image.shape[:2])
    # Create a heatmap
    plt.imshow(entropies_image, cmap='jet',alpha=0.5)
    plt.imshow(segmented_image,alpha=0.5)
    plt.show()

# assume we have a segmented image and the entropies of the pixels
# segmented_image = torch.rand(3, 256, 256)
# entropies = torch.rand(256*256)
for i in range(test_batch_size):
    
    overlay_heatmap(Y_pred[i].cpu().detach().numpy(),e[i][0])


# In[51]:


print(Y_pred.shape)


# In[27]:


from scipy.stats import entropy

def patchwise_entropy(image, patch_size=8):
    h, w = image.shape
    entropy_vals = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            hist = np.histogram(patch, bins=256, range=(0, 255))[0]
            hist = hist / np.sum(hist)
            patch_entropy = entropy(hist, base=2)
            entropy_vals.append(patch_entropy)
    return entropy_vals

# Use the Y_pred[0] as it is, assuming it's already a numpy array
image = Y_pred[0]

# Calculate patchwise entropy
entropy_vals = patchwise_entropy(image)

# Create a heatmap of the entropy values
entropy_map = np.zeros(image.shape)
k = 0
for i in range(0, image.shape[0] - 8 + 1, 8):
    for j in range(0, image.shape[1] - 8 + 1, 8):
        entropy_map[i:i + 8, j:j + 8] = entropy_vals[k]
        k += 1

# Display the entropy map
plt.imshow(entropy_map, cmap='jet')
plt.show()


# In[ ]:





# In[51]:


from scipy.stats import entropy

def patchwise_entropy(segmented_image, patch_size=(16,16)):
    """
    Calculates patchwise entropy of a semantically segmented image
    
    Parameters:
    segmented_image (np.ndarray): The input semantically segmented image
    patch_size (tuple): The size of the patches
    
    Returns:
    np.ndarray: The patchwise entropy map
    """
    # Get the number of classes
    classes = np.unique(segmented_image)
    num_classes = len(classes)
    
    # Divide the image into patches
    patches = divide_into_patches(segmented_image, patch_size)
    
    # Calculate the entropy of each patch
    entropy_map = np.zeros(segmented_image.shape)
    for patch in patches:
        patch_prob = np.zeros(num_classes)
        for c in range(num_classes):
            patch_prob[c] = np.sum(patch == classes[c]) / patch.size
        patch_entropy = entropy(patch_prob, base=2)
        entropy_map[patch.nonzero()] = patch_entropy
    
    return entropy_map

def divide_into_patches(image, patch_size):
    """
    Divides an image into overlapping or non-overlapping patches
    
    Parameters:
    image (np.ndarray): The input image
    patch_size (tuple): The size of the patches
    
    Returns:
    list: A list of patches
    """
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(patch)
    return patches


# In[ ]:





# In[53]:


import torch
import numpy as np
from scipy.stats import entropy

def patchwise_entropy(segmented_image, patch_size):
    """
    Function to calculate patchwise entropy of a semantically segmented image.

    Parameters:
    - segmented_image (torch.Tensor): A tensor of shape (C, H, W) representing the semantically segmented image.
    - patch_size (int): Size of the patch to calculate entropy for.

    Returns:
    - entropy_map (np.ndarray): A 2D numpy array representing the entropy map of the segmented image.
    """
    C, H, W = segmented_image.shape
    entropy_map = np.zeros((H, W))

    # Calculating the histograms for each patch
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = segmented_image[:, i:i + patch_size, j:j + patch_size].flatten()
            histogram, _ = np.histogram(patch, bins=C, range=(0, C), density=True)
            entropy_map[i:i + patch_size, j:j + patch_size] = entropy(histogram, base=2)

    return entropy_map


# In[55]:


def patchwise_entropy(segmented_image, patch_size):
    """
    Function to calculate patchwise entropy of a semantically segmented image.

    Parameters:
    - segmented_image (np.ndarray): A 2D numpy array representing the semantically segmented image.
    - patch_size (int): Size of the patch to calculate entropy for.

    Returns:
    - entropy_map (np.ndarray): A 2D numpy array representing the entropy map of the segmented image.
    """
    H, W = segmented_image.shape
    entropy_map = np.zeros((H, W))

    # Adjust the number of classes (bins) if necessary
    num_classes = np.max(segmented_image) + 1

    # Calculating the histograms for each patch
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = segmented_image[i:i + patch_size, j:j + patch_size].flatten()
            histogram, _ = np.histogram(patch, bins=num_classes, range=(0, num_classes), density=True)
            entropy_map[i:i + patch_size, j:j + patch_size] = entropy(histogram, base=2)

    return entropy_map


# In[165]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming Y_pred is already a numpy array with shape (batch_size, C, H, W)
batch_size = Y_pred.shape[0]

for i in range(batch_size):
    # Extract the i-th image from the batch
    image = Y_pred[i]

    # Calculate entropy map for the i-th image
    entropy_map = patchwise_entropy(image, 8)

    # Plotting the entropy map
    plt.figure(figsize=(4, 4))  # Adjust the size as needed
    plt.imshow(entropy_map, cmap='jet')
    plt.title(f'Entropy Map - Image {i+1}')
    plt.axis('off')
    plt.show()


# In[ ]:





# In[168]:


import matplotlib.pyplot as plt

# Find the maximum entropy value
max_entropy = np.max(entropy_map)

# Get the coordinates of patches with the highest entropy
max_entropy_coords = np.argwhere(entropy_map == max_entropy)

print("Coordinates of patches with the highest entropy:")
for coord in max_entropy_coords:
    print(f"({coord[0]}, {coord[1]})")

# Plot the entropy map
plt.imshow(entropy_map, cmap='jet')
plt.show()


# In[167]:


import matplotlib.pyplot as plt

# Find the maximum entropy value
max_entropy = np.max(entropy_map)

# Get the coordinates of patches with the highest entropy
max_entropy_coords = np.argwhere(entropy_map == max_entropy)

print("Coordinates of patches with the highest entropy:")
for coord in max_entropy_coords:
    print(f"({coord[0]}, {coord[1]})")

# Plot the entropy map
plt.imshow(entropy_map, cmap='jet')
plt.show()


# In[ ]:





# In[ ]:





# # Loop 1 - Depth Map1

# ## MiDAS

# In[95]:


import torch
import cv2
from torchvision.transforms import Compose
import matplotlib.pyplot as plt  # Importing matplotlib

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Choose the right model type
model_type = "DPT_Large"  # Replace with "DPT_Hybrid" or "MiDaS_small" as needed

# Select the appropriate transforms
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Path to your image
filename = '/Users/srirammandalika/Downloads/unlabeled_data/Loop1/raw/0.png'

# Read and process the image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply transforms
input_batch = transform(img).to(device)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)

    # Resize to original image resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

# Convert to numpy array and normalize
output = prediction.cpu().numpy()
output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)

# Apply colormap for visualization
depth_colormap = cv2.applyColorMap(output_normalized.astype('uint8'), cv2.COLORMAP_JET)

# Display depth map using matplotlib
plt.imshow(depth_colormap)
plt.title('Depth Map')
plt.show()


# In[97]:


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the raw image
image_path = '/Users/srirammandalika/Downloads/unlabeled_data/Loop1/raw/0.png'  # Replace with your local path
raw_image = Image.open(image_path).convert('RGB')

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformation to the image
input_tensor = transform(raw_image).unsqueeze(0)  # Add batch dimension

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

# Initialize Grad-CAM (Assuming GradCAM class and generate_heatmap method are defined)
grad_cam = GradCAM(model, 'layer4')

# Generate the heatmap from the model
heatmap = grad_cam.generate_heatmap(input_tensor)

# Normalize and convert heatmap to colormap for visualization
heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
heatmap = np.uint8(255 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min))
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Resize the heatmap to the size of the raw image
colored_heatmap = cv2.resize(colored_heatmap, (raw_image.width, raw_image.height), interpolation=cv2.INTER_AREA)

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas.to(device)
midas.eval()

# Prepare the image for depth estimation
# The transforms for the MiDaS model are designed to be applied to NumPy arrays, not PIL Images
midas_transform = midas_transforms.dpt_transform

# Convert the PIL image to a NumPy array
img_np = np.array(raw_image)

# Apply the MiDaS transforms
input_batch = midas_transform(img_np).to(device)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)

    # Resize to original image resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(raw_image.height, raw_image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

# Convert to numpy array and normalize
depth_map = prediction.cpu().numpy()
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

# Apply colormap for visualization
depth_colormap = cv2.applyColorMap(depth_map_normalized.astype('uint8'), cv2.COLORMAP_JET)

# Overlay the heatmap on the depth map
superimposed_img = cv2.addWeighted(depth_colormap, 0.6, colored_heatmap, 0.4, 0)

# Visualize the result
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()


# In[98]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load your depth map and raw image here
# depth_map = cv2.imread('path_to_depth_map.png', cv2.IMREAD_UNCHANGED)
raw_image = Image.open('/Users/srirammandalika/Downloads/unlabeled_data/Loop1/raw/0.png').convert('RGB')

# Normalize the depth map to have values between 0 and 1
depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Convert raw_image to numpy array if it's a PIL Image
raw_image_np = np.array(raw_image)

# Ensure the raw_image is normalized between 0 and 1
raw_image_np = raw_image_np.astype(np.float32) / 255.0

# Expand the depth map to three channels by replicating the single channel across the RGB channels
depth_map_expanded = np.repeat(depth_map_normalized[:, :, np.newaxis], 3, axis=2)

# Perform the pathwise product
pathwise_product = np.multiply(depth_map_expanded, raw_image_np)

# Display the result
plt.imshow(pathwise_product)
plt.axis('off')  # Hide the axis
plt.show()


# In[118]:


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Assume pathwise_product is the output from the previous code block and is available here

# Convert pathwise_product to PIL Image to apply transformation
pathwise_product_pil = Image.fromarray((pathwise_product * 255).astype('uint8'), 'RGB')

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformation to the pathwise product image
input_tensor = transform(pathwise_product_pil).unsqueeze(0)  # Add batch dimension

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

# Initialize Grad-CAM (Make sure to define GradCAM class or import it if it's from an external library)
grad_cam = GradCAM(model, 'layer4')

# Generate the heatmap from the model
heatmap = grad_cam.generate_heatmap(input_tensor)

# Normalize and convert heatmap to colormap for visualization
heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
heatmap = np.uint8(255 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min))
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Invert the colors of the heatmap
colored_heatmap = colored_heatmap[:, :, ::-1]

# Resize the heatmap to the size of the pathwise product image
colored_heatmap = cv2.resize(colored_heatmap, (pathwise_product_pil.width, pathwise_product_pil.height), interpolation=cv2.INTER_AREA)

# Overlay the heatmap on the pathwise product image
superimposed_img = cv2.addWeighted(np.array(pathwise_product_pil), 0.6, colored_heatmap, 0.4, 0)

# Visualize the result
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()


# In[119]:


# Ensure both images are of the same size
entropy_map_resized = cv2.resize(entropy_map, (superimposed_img.shape[1], superimposed_img.shape[0]))

# Normalize the entropy map to be between 0 and 255
entropy_map_normalized = cv2.normalize(entropy_map_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Convert the normalized entropy map to a colormap
colored_entropy_map = cv2.applyColorMap(entropy_map_normalized, cv2.COLORMAP_JET)

# Overlay the entropy map on the Grad-CAM image
fused_image = cv2.addWeighted(superimposed_img, 0.6, colored_entropy_map, 0.4, 0)

# Visualize the result
plt.imshow(fused_image)
plt.axis('off')
plt.show()


# In[ ]:





# ## Oracle - Loop1

# In[127]:


import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(test_batch_size, 4, figsize=(4*5, test_batch_size*5))

for i in range(test_batch_size):
    # Convert tensors to numpy arrays
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i]  # Removed .cpu() as Y_pred[i] is already a numpy array

    # Calculate entropy for each patch in the predicted label class
    entropy_vals = patchwise_entropy(label_class_predicted)

    # Determine the threshold for high entropy - this is a parameter you might need to adjust
    high_entropy_threshold = np.percentile(entropy_vals, 90)    # For example, use the 90th percentile

    # Apply the threshold to create a binary mask for high-entropy regions
    high_entropy_mask = np.zeros(label_class_predicted.shape, dtype=bool)
    k = 0
    for x in range(0, label_class_predicted.shape[0], 8):
        for y in range(0, label_class_predicted.shape[1], 8):
            high_entropy_mask[x:x+8, y:y+8] = entropy_vals[k] > high_entropy_threshold
            k += 1

    # Replace high-entropy regions in the predicted label class with actual label class regions
    modified_segmented_image = np.copy(label_class_predicted)
    modified_segmented_image[high_entropy_mask] = label_class[high_entropy_mask]

    # Plot the images
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")
    axes[i, 3].imshow(modified_segmented_image)
    axes[i, 3].set_title("Modified Segmented Image")

plt.tight_layout()
plt.show()


# In[123]:


import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(test_batch_size, 4, figsize=(4*5, test_batch_size*5))

for i in range(test_batch_size):
    # Assuming you have the images in the correct format
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i]  # Directly use Y_pred[i] as it's already a numpy array

    # Calculate entropy and create an entropy map for the current segmented image
    entropy_vals = patchwise_entropy(label_class_predicted)
    entropy_map = np.zeros(label_class_predicted.shape)
    k = 0
    for x in range(0, label_class_predicted.shape[0] - 8 + 1, 8):
        for y in range(0, label_class_predicted.shape[1] - 8 + 1, 8):
            entropy_map[x:x + 8, y:y + 8] = entropy_vals[k]
            k += 1

    # Define high entropy threshold
    high_entropy_threshold = np.mean(entropy_vals) + np.std(entropy_vals)

    # Create a mask for high entropy patches
    high_entropy_mask = entropy_map > high_entropy_threshold

    # Replace high entropy patches in label_class_predicted with corresponding patches from label_class
    modified_segmented_image = np.copy(label_class_predicted)
    for x in range(0, label_class_predicted.shape[0] - 8 + 1, 8):
        for y in range(0, label_class_predicted.shape[1] - 8 + 1, 8):
            if high_entropy_mask[x:x + 8, y:y + 8].any():  # Check if any part of the patch is high entropy
                modified_segmented_image[x:x + 8, y:y + 8] = label_class[x:x + 8, y:y + 8]

    # Plotting
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")
    axes[i, 3].imshow(modified_segmented_image)
    axes[i, 3].set_title("Modified Segmented Image")

plt.tight_layout()
plt.show()


# In[ ]:





# In[134]:


import torchvision.transforms.functional as TF
from PIL import Image

# Define the path where the images will be saved
save_path = '/Users/srirammandalika/Downloads/unlabeled_data/Loop1/label'  # Replace with your actual path

# Loop over the batch and save the images
for i in range(test_batch_size):
    # Process the landscape image
    landscape_img = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    # When the image is float and in the range [0, 1], mode should be 'RGB'
    landscape_img_pil = TF.to_pil_image(landscape_img, mode='RGB')
    landscape_img_pil = TF.resize(landscape_img_pil, [256, 256])
    landscape_img_pil.save(f"{save_path}/landscape_{i}.png")  # Ensure there's a slash before the filename
    
    # Process the label class image
    # Assuming label_class_img is a single-channel image, hence mode='L' for grayscale
    label_class_img = Y[i].cpu().detach().numpy().astype('uint8')  # Convert to unsigned 8-bit integer
    label_class_img_pil = Image.fromarray(label_class_img, mode='L')  # Create PIL image from numpy array
    label_class_img_pil = TF.resize(label_class_img_pil, [256, 256])
    label_class_img_pil.save(f"{save_path}/label_class_{i}.png")  # Ensure there's a slash before the filename

print("Images have been saved.")


# In[136]:


import numpy as np
from PIL import Image

# Define the path where the images will be saved
save_path = '/Users/srirammandalika/Downloads/unlabeled_data/Loop1/label'  # Replace with your local directory path

# Loop over the batch and save the label class images
for i in range(test_batch_size):
    # Process the label class image
    label_class_img = Y[i].cpu().detach().numpy()
    # Convert to 8-bit image if it's not already
    label_class_img_pil = Image.fromarray(label_class_img.astype(np.uint8))
    # Save the image
    label_class_img_pil.save(f"{save_path}label_class_{i}.png")

print("Label class images have been saved.")


# In[ ]:





# In[47]:


print(segmented_image.shape)


# In[48]:


print(Y_pred.shape)


# In[49]:


print(entropy_map.shape)


# In[ ]:





# In[ ]:





# # Making Local Dataset

# In[50]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where you want to save the images
output_directory = "/Users/srirammandalika/Downloads/U-Net Output"
os.makedirs(output_directory, exist_ok=True)

for i in range(test_batch_size):
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    # Save the "Label Class - Predicted" image
    image_filename = os.path.join(output_directory, f"image_{i}.png")
    cv2.imwrite(image_filename, label_class_predicted * 255)  # Convert to 0-255 scale
    
    # Optional: Display the saved image
    plt.imshow(label_class_predicted)
    plt.title("Label Class - Predicted")
    plt.show()


# In[51]:


import os
import matplotlib.pyplot as plt

# Define your variables
test_batch_size = 8
# Define other variables like X, Y_pred, etc.

# Create an output directory
output_dir = "/Users/srirammandalika/Downloads/U-Net Output"
os.makedirs(output_dir, exist_ok=True)

# Loop through predicted images, save as PNG files
for i in range(test_batch_size):
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    image_path = os.path.join(output_dir, f"predicted_image_{i}.png")
    plt.imsave(image_path, label_class_predicted)


# In[52]:


import os
import matplotlib.pyplot as plt
import shutil

# Define your variables
test_batch_size = 8


# Create an output directory
output_dir = "/Users/srirammandalika/Downloads/U-NetOutput"
os.makedirs(output_dir, exist_ok=True)

# Loop through predicted images, save original images, and create a zip archive
for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    image_path = os.path.join(output_dir, f"original_image_predicted_{i}.png")
    plt.imsave(image_path, landscape)

# Create a zip archive of the output directory
shutil.make_archive(output_dir, 'zip', output_dir)


# In[ ]:





# # Annotation Section

# In[53]:


#Use Label Studio for this Section


# # Re-Loop Section

# In[54]:


import torch

# Define the device (CPU or GPU) based on your system configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize your UNet model
model = UNet(num_classes=num_classes).to(device)

# Load the pretrained weights
model.load_state_dict(torch.load("/Users/srirammandalika/Desktop/U-Net.pth", map_location=device))
model.eval()  # Set the model to evaluation mode


# In[ ]:





# In[ ]:





# In[128]:


import os
from PIL import Image

# Specify the paths to the image and label class directories
image_dir = '/Users/srirammandalika/Downloads/LocalCityscapesDataset_/Images'  # Update with your image directory
label_class_dir = '/Users/srirammandalika/Downloads/LocalCityscapesDataset_/Labels'  # Update with your label class directory

# List all image files in the image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Create a subplot for displaying the images and their corresponding label class images
fig, axes = plt.subplots(len(image_files), 2, figsize=(5 * 2, 5 * len(image_files)))

# Sort the image files to match corresponding label class files
image_files.sort()

# Plot each image with its corresponding label class image
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    label_file = image_file.replace('.png', '_label.png')  # Derive the label file name
    label_path = os.path.join(label_class_dir, label_file)
    
    image = Image.open(image_path)
    Nlabel_class = Image.open(label_path)
    
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Image {i}")
    
    axes[i, 1].imshow(Nlabel_class)
    axes[i, 1].set_title(f"Label Class {i}")

plt.show()



# In[129]:


import os
from PIL import Image
from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib.pyplot as plt

# Specify the paths to the directories
label_class_dir = '/Users/srirammandalika/Downloads/LocalCityscapesDataset_/Labels'  # Update with your label class directory
predicted_dir = '/Users/srirammandalika/Downloads/U-Net Output'  # Update with your predicted directory

# List all label class files in the label class directory
label_files = [f for f in os.listdir(label_class_dir) if f.endswith('_label.png')]

# Sort the file names to ensure they are processed in the correct order
label_files.sort()

# Create a subplot for displaying the images
fig, axes = plt.subplots(len(label_files), 3, figsize=(5 * 3, 5 * len(label_files)))

# Initialize a list to store IoU values
iou_values = []

# Calculate IoU for each image
for i, label_file in enumerate(label_files):
    # Load label class and predicted images
    label_path = os.path.join(label_class_dir, label_file)
    predicted_file = label_file.replace('_label.png', '.png')  # Derive the predicted file name
    predicted_path = os.path.join(predicted_dir, predicted_file)

    label_class = np.array(Image.open(label_path))
    predicted_segmentation = np.array(Image.open(predicted_path))

    # Flatten the images for IoU calculation
    label_class_flat = label_class.flatten()
    predicted_flat = predicted_segmentation.flatten()

    # Ensure both arrays have the same number of elements
    min_len = min(len(label_class_flat), len(predicted_flat))
    label_class_flat = label_class_flat[:min_len]
    predicted_flat = predicted_flat[:min_len]

    # Calculate IoU for the current image
    iou = jaccard_score(label_class_flat, predicted_flat, average='micro')  # Use 'micro', 'macro', 'weighted', or None
    iou_values.append(iou)

    # Plot images
    axes[i, 0].imshow(label_class)
    axes[i, 0].set_title(f"Label Class {i}")

    axes[i, 1].imshow(predicted_segmentation)
    axes[i, 1].set_title(f"Predicted Segmentation {i}")

    axes[i, 2].imshow(label_class)  # You can customize this to show the difference or overlay
    axes[i, 2].imshow(predicted_segmentation, alpha=0.5, cmap='viridis')  # Overlay with transparency
    axes[i, 2].set_title(f"Overlay {i}")

    #print(f"IoU for Image {i}: {iou:.4f}")

plt.show()

# The list 'iou_values' contains the IoU values for each image
#print("IoU values:", iou_values)


# #  Retrain using Labeled Pool

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
import os

# Directories
image_dir = '/Users/srirammandalika/Downloads/LocalCityscapesDataset_/Images'
label_dir = '/Users/srirammandalika/Downloads/LocalCityscapesDataset_/Labels'

# Get all image file names
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

# Plotting
fig, axes = plt.subplots(len(image_files), 2, figsize=(10, len(image_files) * 5))

for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    # Load images
    img = Image.open(os.path.join(image_dir, image_file))
    label = Image.open(os.path.join(label_dir, label_file))

    # Plot image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image: {image_file}")
    axes[i, 0].axis('off')

    # Plot label
    axes[i, 1].imshow(label)
    axes[i, 1].set_title(f"Label: {label_file}")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()


# # Image Splitter

# # Attempt 2 - Label Retrain

# In[130]:


import matplotlib.pyplot as plt
from PIL import Image
import os

# Directories
image_dir = '/Users/srirammandalika/Downloads/unlabeled_data/Loop3/raw'
label_dir = '/Users/srirammandalika/Downloads/unlabeled_data/Loop3/label'

# Get all image file names
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:32]
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])[:32]

# Ensure that the number of images and labels are the same
num_files = min(len(image_files), len(label_files))

# Plotting
fig, axes = plt.subplots(num_files, 2, figsize=(10, num_files * 5))

for i in range(num_files):
    # Load images
    img = Image.open(os.path.join(image_dir, image_files[i]))
    label = Image.open(os.path.join(label_dir, label_files[i]))

    # Plot image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image: {image_files[i]}")
    axes[i, 0].axis('off')

    # Plot label
    axes[i, 1].imshow(label)
    axes[i, 1].set_title(f"Label: {label_files[i]}")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx]) # Assuming label names are same as images
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # L for grayscale, assuming your labels are in grayscale format

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            label = torch.squeeze(label, 0)  # Remove the channel dimension
            label = label.long()  # Convert label to long tensor, required for CrossEntropyLoss

        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure the input size is consistent
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Labels must be resized to match the input size
    transforms.ToTensor()
])


# Dataset and DataLoader
image_dir = '/Users/srirammandalika/Downloads/unlabeled_data/raw'
label_dir = '/Users/srirammandalika/Downloads/unlabeled_data/label'
dataset = CustomDataset(image_dir, label_dir, transform, target_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Adjust batch size as necessary

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/Users/srirammandalika/Desktop/U-Net-v1.pth'
num_classes = num_classes
model = UNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adjust learning rate as necessary
criterion = torch.nn.CrossEntropyLoss() # Or another appropriate loss function

for epoch in range(50):  # You mentioned 30 epochs in the comment, but the loop is set for 50
    epoch_loss = 0
    for inputs, labels in dataloader:
        # Transfer Data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the Loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        epoch_loss += loss.item()
    
    # Compute the average loss for the epoch
    epoch_loss /= len(dataloader)
    print(f'Epoch [{epoch+1}/50], Average Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), '/Users/srirammandalika/Desktop/U-Net-v2.pth')


# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.target_transform:
            label = self.target_transform(label)
            label = torch.tensor(np.array(label), dtype=torch.long)

        # Debugging: Check for out-of-range values
        if label.max() >= num_classes:
            print(f"Invalid label value {label.max()} found in file {self.images[idx]}")
        
        return image, label



# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the U-Net input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Target transformations
target_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize labels to match the U-Net input
    # Do not convert labels to tensor here; it's handled in __getitem__
])


# Dataset and DataLoader
image_dir = '/Users/srirammandalika/Downloads/unlabeled_data/raw'
label_dir = '/Users/srirammandalika/Downloads/unlabeled_data/label'
dataset = CustomDataset(image_dir, label_dir, transform, target_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Adjust batch size as necessary

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/Users/srirammandalika/Desktop/U-Net-v1.pth'  # Replace with your model path
model = UNet(num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.train()

# Training
# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    epoch_loss = 0
    model.train()  # Set model to training mode
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Average loss for the epoch
    epoch_loss /= len(dataloader)
    print(f'Epoch [{epoch+1}/50], Average Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), '/Users/srirammandalika/Desktop/U-Net-v2.pth')


# In[ ]:


get_ipython().run_line_magic('cd', '')


# ## IoU

# In[ ]:


import torch
from torchvision.transforms import ToPILImage, Compose, Resize
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
# Assuming RawImageDataset and LabelDataset are defined elsewhere in your script

def calculate_iou(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Transformation to resize images
resize_transform = Compose([Resize((256, 256))])

# Load and process the images and labels
raw_data_dir = '/Users/srirammandalika/Downloads/unlabeled_data/Loop3/raw'
label_data_dir = '/Users/srirammandalika/Downloads/unlabeled_data/Loop3/label'

raw_dataset = RawImageDataset(raw_data_dir, transform=Compose([resize_transform, transform]))
label_dataset = LabelDataset(label_data_dir)  # Removed transform argument

raw_loader = DataLoader(raw_dataset, batch_size=16, shuffle=False)
label_loader = DataLoader(label_dataset, batch_size=16, shuffle=False)

iou_values = []

to_pil_image = ToPILImage()  # Ensure this is correctly imported

for (images, labels) in zip(raw_loader, label_loader):
    images = images.to(device)
    labels = labels.cpu().numpy()

    with torch.no_grad():
        outputs = model(images)

    predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

    for i in range(len(images)):
        # Ensure labels[i] is in the correct shape
        label_img = labels[i]
        if label_img.ndim == 2:  # Check if it's a 2D array
            label_img = np.expand_dims(label_img, axis=0)  # Add a channel dimension

        # Convert to PIL image and resize
        label_img_tensor = torch.from_numpy(label_img)  # Convert numpy array to tensor
        label_resized = resize_transform(ToPILImage()(label_img_tensor))

        # Convert back to single-channel numpy array
        label_resized_np = np.array(label_resized)[:, :, 0]  # Selecting only one channel

        iou_score = calculate_iou(predicted_labels[i], label_resized_np)
        iou_values.append(iou_score)

        # Visualization
        plt.figure(figsize=(12, 4))

        # Original Image
        plt.subplot(1, 3, 1)
        original_image = to_pil_image(images[i].cpu()).convert("RGB")
        plt.imshow(np.array(original_image))
        plt.title("Original Image")
        plt.axis('off')

        # Predicted Segmentation
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_labels[i], cmap='viridis')
        plt.title("Predicted Segmentation")
        plt.axis('off')

        # Label Image
        plt.subplot(1, 3, 3)
        label_image = label_resized.convert("RGB")
        plt.imshow(np.array(label_image))
        plt.title("Label Image")
        plt.axis('off')

        plt.show()

# Calculate and print Peak and Average IoU
peak_iou = np.max(iou_values)
avg_iou = np.mean(iou_values)
print(f"Peak IoU: {peak_iou}")
print(f"Average IoU: {avg_iou}")


# In[ ]:


# Calculate and print Peak and Average IoU
peak_iou = np.max(iou_values)
avg_iou = np.mean(iou_values)
print(f"Peak IoU: {peak_iou}")
print(f"Average IoU: {avg_iou}")


# In[56]:


import torch
import cv2
from torchvision.transforms import Compose
import matplotlib.pyplot as plt  # Importing matplotlib

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Choose the right model type
model_type = "DPT_Large"  # Replace with "DPT_Hybrid" or "MiDaS_small" as needed

# Select the appropriate transforms
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Path to your image
filename = '/Users/srirammandalika/Desktop/Sample.png'

# Read and process the image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply transforms
input_batch = transform(img).to(device)

# Perform depth estimation
with torch.no_grad():
    prediction = midas(input_batch)

    # Resize to original image resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

# Convert to numpy array and normalize
output = prediction.cpu().numpy()
output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)

# Apply colormap for visualization
depth_colormap = cv2.applyColorMap(output_normalized.astype('uint8'), cv2.COLORMAP_JET)

# Display depth map using matplotlib
plt.imshow(depth_colormap)
plt.title('Depth Map')
plt.show()


# In[74]:


import torchvision.models as models

# Example: Load a pre-trained model (like ResNet)
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()


# In[91]:


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the raw image
image_path = '/Users/srirammandalika/Downloads/unlabeled_data/Loop3/raw/142.png'
raw_image = Image.open(image_path).convert('RGB')

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformation to the image
input_tensor = transform(raw_image).unsqueeze(0)  # Add batch dimension

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

# Initialize Grad-CAM
grad_cam = GradCAM(model, 'layer4')

# Generate the heatmap from the model
heatmap = grad_cam.generate_heatmap(input_tensor)

# Normalize and convert heatmap to colormap for visualization
heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
heatmap = np.uint8(255 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min))
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Resize the heatmap to the size of the raw image
colored_heatmap = cv2.resize(colored_heatmap, (raw_image.width, raw_image.height), interpolation=cv2.INTER_AREA)

# Convert raw image to numpy array
raw_image_np = np.array(raw_image)

# Overlay the heatmap on the raw image
superimposed_img = cv2.addWeighted(raw_image_np, 0.6, colored_heatmap, 0.4, 0)

# Visualize the result
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()


# In[ ]:





# In[93]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load your depth map and raw image here
# depth_map = cv2.imread('path_to_depth_map.png', cv2.IMREAD_UNCHANGED)
raw_image = Image.open('/Users/srirammandalika/Downloads/unlabeled_data/Loop3/raw/142.png').convert('RGB')

# Normalize the depth map to have values between 0 and 1
depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# Convert raw_image to numpy array if it's a PIL Image
raw_image_np = np.array(raw_image)

# Ensure the raw_image is normalized between 0 and 1
raw_image_np = raw_image_np.astype(np.float32) / 255.0

# Expand the depth map to three channels by replicating the single channel across the RGB channels
depth_map_expanded = np.repeat(depth_map_normalized[:, :, np.newaxis], 3, axis=2)

# Perform the pathwise product
pathwise_product = np.multiply(depth_map_expanded, raw_image_np)

# Display the result
plt.imshow(pathwise_product)
plt.axis('off')  # Hide the axis
plt.show()


# In[99]:


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Assume pathwise_product is the output from the previous code block and is available here

# Convert pathwise_product to PIL Image to apply transformation
pathwise_product_pil = Image.fromarray((pathwise_product * 255).astype('uint8'), 'RGB')

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformation to the pathwise product image
input_tensor = transform(pathwise_product_pil).unsqueeze(0)  # Add batch dimension

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

# Initialize Grad-CAM (Make sure to define GradCAM class or import it if it's from an external library)
grad_cam = GradCAM(model, 'layer4')

# Generate the heatmap from the model
heatmap = grad_cam.generate_heatmap(input_tensor)

# Normalize and convert heatmap to colormap for visualization
heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
heatmap = np.uint8(255 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min))
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Resize the heatmap to the size of the pathwise product image
colored_heatmap = cv2.resize(colored_heatmap, (pathwise_product_pil.width, pathwise_product_pil.height), interpolation=cv2.INTER_AREA)

# Overlay the heatmap on the pathwise product image
superimposed_img = cv2.addWeighted(np.array(pathwise_product_pil), 0.6, colored_heatmap, 0.4, 0)

# Visualize the result
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()

