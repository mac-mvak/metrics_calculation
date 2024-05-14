import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image


def gram(A):
    return (A @ A.T) / A.shape[1]

def style_score(img1_path, img2_path):
    image1 = Image.open(img1_path).convert('RGB')
    image2 = Image.open(img2_path).convert('RGB')
    image1_tensor = transforms.ToTensor()(image1).flatten(1)
    image2_tensor = transforms.ToTensor()(image2).flatten(1)


    ans = torch.square(gram(image1_tensor) - gram(image2_tensor)).mean()
    return ans.item()


img1_path = 'runs_gaussian/test_FT_IMAGENET_cezanne_s512_t603_ninv40_ngen6_dir_6.0_l1_10.0_st_1.0_lr_4e-06_gaussian/image_samples/style_0_orig.png'
img2_path = 'runs_gaussian/test_FT_IMAGENET_cezanne_s512_t603_ninv40_ngen6_dir_6.0_l1_10.0_st_1.0_lr_4e-06_gaussian/image_samples/train_49_0_orig_color.png'
style_score(img1_path, img2_path)


