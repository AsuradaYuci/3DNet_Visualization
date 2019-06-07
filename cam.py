#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @time:2019/6/5下午3:11
# @Author: Yu Ci

import torch
import torch.nn.functional as F


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        # 注册一个hook来保存激活和渐变的值
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        Args:
            model: a base model to get CAM which have global pooling and fully connected layer.
            target_layer: conv_layer before Global Average Pooling
        """

        self.model = model  # 模型
        self.target_layer = target_layer  # 目标卷积层

        # save values of activations and gradients in target_layer  目标卷积层的激活值和梯度值
        self.values = SaveValues(self.target_layer)

    def forward(self, x, flows, targets):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)  输入图片
        Return:
            heatmap: class activation mappings of the predicted class  激活图
        """

        # object classification 目标分类
        score = self.model(x)
        prob = F.softmax(score, dim=1)
        max_prob, idx = torch.max(prob, dim=1)
        print(
            "predicted object ids {}\t probability {}".format(idx.item(), max_prob.item()))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data
        cam = self.getCAM(self.values, weight_fc, idx.item())

        return cam

    def __call__(self, x, flows, targets):
        return self.forward(x, flows, targets)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer, regular_criterion):
        super().__init__(model, target_layer)
        self.regular_criterion = regular_criterion
        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, flows, targets):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class
        """
        # object classification
        feat, feat_raw = self.model(x, flows)
        featsize = feat.size()  # 8,8,128
        featbatch = featsize[0]
        seqlen = featsize[1]

        # expand the target label ID loss
        featX = feat.view(featbatch * seqlen, -1)  # 64,128

        targetX = targets.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
        targetX = targetX.expand(featbatch, seqlen)
        # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
        targetX = targetX.contiguous()
        targetX = targetX.view(featbatch * seqlen, -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
        targetX = targetX.squeeze(1)

        loss_id, outputs_id = self.regular_criterion(featX, targetX)
        score = outputs_id  # torch.Size([64, 150])  相当于线性分类

        prob = F.softmax(score, dim=1)  # torch.Size([1, 1000])                      torch.Size([64, 150])
        max_prob, idx = torch.max(prob, dim=1)  # torch.Size([64])
        # print("predicted object ids {}\t probability {}".format(idx.item(), max_prob.item()))
        length = idx.size()[0]
        cam_final = list()

        for i in range(length):
            index = idx.cpu().numpy()[i]
        # caluculate cam of the predicted class
            score_i = score[i].unsqueeze(0)
            cam = self.getGradCAM(self.values, score_i, index.item(), i)
            cam_final.append(cam)

        return cam_final

    def __call__(self, x, flows, targets):
        return self.forward(x, flows, targets)

    def getGradCAM(self, values, score, idx, i):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()
        score[0, idx].backward(retain_graph=True)
        activations = values.activations[i].unsqueeze(0)  # torch.Size([64, 2048, 8, 4])
        gradients = values.gradients[i].unsqueeze(0)
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(GradCAM):
    """ Grad CAM plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, flows):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x, flows)  # torch.Size([1, 1000])
        prob = F.softmax(score, dim=1)  # torch.Size([1, 1000])
        max_prob, idx = torch.max(prob, dim=1)
        idx = idx.cpu().numpy()[0]
        print(
            "predicted object ids {}\t probability {}".format(idx.item(), max_prob.item()))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx.item())

        return cam

    def __call__(self, x, flows):
        return self.forward(x, flows)

    def getGradCAMpp(self, values, score, index):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()
        score[0, index].backward(retain_graph=True)
        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, index].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data
