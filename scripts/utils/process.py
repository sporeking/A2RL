from skimage.metrics import structural_similarity as ssim
from utils import device
import numpy
import cv2
import torch
import torch.nn.functional as F

def contrast_ssim(img1, img2, win_size=7):
    if img1 is None or img2 is None:
        return 0
    
    target_size = (max(min(img1.shape[0], img2.shape[0]), 15),
                    max(min(img1.shape[1], img2.shape[1]), 15))
    target_size = (target_size[1], target_size[0]) # cv2.resize is (width, height)
    if img1.shape != target_size:
        img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_LINEAR)
    if img2.shape != target_size:
        img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_LINEAR)

    # print(img1.shape, img2.shape)
    return ssim(img1, img2, channel_axis=-1, win_size=7)

def contrast_ssim_resize(img1, img2, win_size=7):
    if img1 is None or img2 is None:
        return 0

    if (img1.shape[0] < 10 and img1.shape[1] < 10) or (img2.shape[0] < 10 and img2.shape[1] < 10):
        return 0
    
    target_size = (30, 30)
    img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_LINEAR)

    return ssim(img1, img2, channel_axis=-1, win_size=7)

def contrast_hist(mutation1, mutation2) -> float: 
    if mutation1 is None or mutation2 is None:
        return 0
    if mutation1.shape != mutation2.shape:
        target_size = (min(mutation1.shape[0], mutation2.shape[0]),
                       min(mutation1.shape[1], mutation2.shape[1]))
        mutation1 = cv2.resize(mutation1, target_size, interpolation=cv2.INTER_AREA)
        mutation2 = cv2.resize(mutation2, target_size, interpolation=cv2.INTER_AREA)
    hist_1, hist_2 = cv2.calcHist([mutation1], [0], None, [256], [0, 256]), cv2.calcHist([mutation2], [0], None, [256], [0, 256])
    hist_1, hist_2 = cv2.normalize(hist_1, hist_1).flatten(), cv2.normalize(hist_2, hist_2).flatten()
    correlation = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
    similar = abs(correlation)
    return similar

def contrast(mutation1, mutation2) -> float: # that means the similarity between two mutations
    if mutation1 is None or mutation2 is None:
        return 0
    return ssim(mutation1, mutation2, multichannel=True, channel_axis=2)

        
def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    return input_tensor

def ssim_gpu(img1, img2, win_size=7):
    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(2)
        img2 = img2.unsqueeze(2)
    
    img1 = img1.float()
    img2 = img2.float()
    
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    window = window.unsqueeze(0) * window.unsqueeze(1)
    window = window.expand(img1.size(1), 1, win_size, win_size)
    
    mu1 = F.conv2d(img1, window, padding=win_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=win_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=win_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=win_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=win_size//2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

