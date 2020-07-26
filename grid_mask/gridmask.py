# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb 12 13:08:02 2020
#
# @author: wuyunpeng
# """
import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        #if np.random.rand() > self.prob:  #-1.96 - 1.96
           # return img
        size = img.shape
        h = size[1]
        w = size[2]
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(self.d1, self.d2)  #这个方法产生离散均匀分布的整数，这些整数大于等于low，小于high。
        #d = self.d
        self.l = int(d*self.ratio+0.5)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, ww//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, ww), 0)
                t = max(min(t, ww), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1-mask

        #mask = mask.expand_as(img)
       # mask = torch.cat([mask,mask],1)
        #mask = mask.reshape(3,800,800)

        img0 = img[0, :, :] * mask  #3 800 800
        img1 = img[1, :, :] * mask  # 3 800 800
        img2 = img[2, :, :] * mask  # 3 800 800
        img = torch.cat([img0,img1,img2],0)
        img = img.reshape(3,h,w)
        return img, mask

class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x

        #c,h,w = x.size()

        y,mask = self.grid(x)
        #y = torch.cat(y).view(c,h,w)
        return y,mask
grid = GridMask(96,224, 360, 0.6,1,0.8)
grid.set_prob(1, 1)
raw_data = './'
save_dir='./result/'
raw_images_dir = os.path.join(raw_data, 'images')

images = [i for i in os.listdir('./images') if 'jpg' in i]

for idx, img in enumerate(images):
    print(idx, 'read image', img)
    image=Image.open(os.path.join(raw_images_dir, img)).convert('RGB')
    name = img.split('.')[0]
    trans=transforms.Compose([transforms.ToTensor()])
    a=trans(image)
    #image = cv2.imread('./1.png')
    input,mask = grid(a)
  #######################################  
    b=np.array(input)
    maxi=b.max()
    b=b*255./maxi
    b=b.transpose(1,2,0).astype(np.uint8)
    xx=Image.fromarray(b).convert('RGB')
    cv = cv2.cvtColor(np.array(xx), cv2.COLOR_RGB2BGR)
    plt.axis('off')
    img = os.path.join(save_dir, "%s_gr.jpg" % (name))  
    cv2.imwrite(img, cv)
############################################################
    mask=np.array(mask)
    maxi=mask.max()
    mask=mask*255./maxi
    #mask=mask.transpose(1,2,0).astype(np.uint8)
    mask=Image.fromarray(mask).convert('RGB')
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    plt.axis('off')
    img = os.path.join(save_dir, "%s_gr_mask.jpg" % (name)) 
    cv2.imwrite(img, mask)
    #############################################



    # plt.imshow(xx)
    # plt.show()
    #plt.savefig('./result/'+ name +'.jpg', bbox_inches='tight')

