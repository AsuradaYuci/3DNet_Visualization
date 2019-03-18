#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-17 13:00
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : action_recognition.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import numpy as np
import cv2
from utils import center_crop
from scipy.ndimage import zoom

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

class ActionRecognition(object):
    def __init__(self, model):
        self.model = model

    def img_process(self, imgs, frames_num):
        images = np.zeros((frames_num, 224, 224, 3))
        orig_imgs = np.zeros_like(images)
        for i in range(frames_num):
            next_image = imgs[i]
            next_image = np.uint8(next_image)
            scaled_img = cv2.resize(next_image, (256, 256), interpolation=cv2.INTER_LINEAR)  # resize to 256x256
            cropped_img = center_crop(scaled_img)  # center crop 224x224
            final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            images[i] = final_img
            orig_imgs[i] = cropped_img
        torch_imgs = torch.from_numpy(images.transpose(3, 0, 1, 2))
        torch_imgs = torch_imgs.float() / 255.0
        mean_3d = [124 / 255, 117 / 255, 104 / 255]
        std_3d = [0.229, 0.224, 0.225]
        for t, m, s in zip(torch_imgs, mean_3d, std_3d):
            t.sub_(m).div_(s)
        return np.expand_dims(orig_imgs, 0), torch_imgs.unsqueeze(0)

    def recognition_video(self, imgs):
        """
        recognition video's action
        :param imgs: preprocess imgs
        :return:
        """
        prediction, _ = self.model(torch.tensor(imgs).cuda())  # 1x101
        pred = torch.argmax(prediction).item()
        return pred

    def generate_cam(self, imgs):
        predictions, layerout = self.model(torch.tensor(imgs).cuda())  # 1x101
        layerout = torch.tensor(layerout[0].numpy().transpose(1, 2, 3, 0))  # 8x7x7x768
        pred_weights = self.model.module.classifier.weight.data.detach().cpu().numpy().transpose()  # 768 x 101
        predictions = torch.nn.Softmax(dim=1)(predictions)
        pred_top3 = predictions.detach().cpu().numpy().argsort()[0][::-1][:3]
        probality_top3 = -np.sort(-predictions.detach().cpu().numpy())[0,0:3]

        #print(pred_top3)
        #pred_top3 = torch.argmax(predictions).item()
        cam_list = list()
        for k in range(len(pred_top3)):
            cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
            for i, w in enumerate(pred_weights[:, pred_top3[k]]):
                # Compute cam for every kernel
                cam += w * layerout[:, :, :, i]  # 8x7x7

            # Resize CAM to frame level
            cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

            # normalize
            cam -= np.min(cam)
            cam /= np.max(cam) - np.min(cam)
            cam_list.append(cam)
        return cam_list, pred_top3, probality_top3
