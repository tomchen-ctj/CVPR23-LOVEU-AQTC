import torch

path = '/home/hy/ssd1/tomchen/loveu2023/encoder/egovlp/egovlp.pth'
# path = 'results/EgoClip_SE_1_mid_scale_th_v2/models/0502_01/checkpoint-epoch4.pth'
checkpoint = torch.load(path)
state_dict = checkpoint['state_dict']