import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import pdb
import torch.nn.functional as F

def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2

            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask




def grad_rollout_batch(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1)).unsqueeze(0).repeat(attentions[0].size(0),1,1).to(attentions[0].device)
    # result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat.scatter_(-1, indices, torch.zeros(indices.shape).cuda())
            I = torch.eye(attention_heads_fused.size(-1)).unsqueeze(0).repeat(attention_heads_fused.size(0),1,1).to(attention_heads_fused[0].device)
            a = (attention_heads_fused + 1.0*I)/2
            # a = a / a.sum(dim=-1)
            a = a / (a.sum(dim=-1)[:,np.newaxis])
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches

    mask = result[:, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(mask.size(0),width, width).cpu().numpy()
    # mask = mask / np.max(mask)
    max_div = np.max(mask,axis=(1,2))
    max_div = np.repeat(np.repeat(max_div[:,np.newaxis],mask.shape[1],axis=1)[:,:,np.newaxis],mask.shape[2],axis=2)
    mask = mask/(max_div+1e-8)
    return mask




class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def clear_cache(self):
        self.attentions = []
        self.attention_gradients = []
    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        category_mask = category_mask.to(output.device)

        loss = (output*category_mask).sum()
        loss.backward()
        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)


class VITAttentionGradRollout_Batch:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.hook_handles = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                self.hook_handles.append(module.register_forward_hook(self.get_attention))
                self.hook_handles.append(module.register_backward_hook(self.get_attention_gradient))

        self.attentions = []
        self.attention_gradients = []


    def remove_hooks(self):
        for i in range(len(self.hook_handles)):
            self.hook_handles[i].remove()

    def clear_cache(self):
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        # self.attentions.append(output.cpu())
        self.attentions.append(output)

    def get_attention_gradient(self, module, grad_input, grad_output):
        # self.attention_gradients.append(grad_input[0].cpu())
        self.attention_gradients.append(grad_input[0])

    # def __call__(self, input_tensor, category_indices):
    def __call__(self, input_tensor, top=True):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.data.topk(1, dim=1)[1][0]
        # category_mask = torch.zeros(output.size())
        # category_mask[:, category_index] = 1
        # category_mask = F.one_hot(category_indices,num_classes =output.size(1))
        if top:
            category_mask = F.one_hot(class_idx,num_classes =output.size(1))
        else:
            raise NotImplementedError
        category_mask = category_mask.to(output.device)
        loss = (output*category_mask).sum()
        loss.backward()
        return grad_rollout_batch(self.attentions, self.attention_gradients,
            self.discard_ratio)
