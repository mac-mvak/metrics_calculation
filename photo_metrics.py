import torch
import torchvision.transforms.functional as TF



from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

SSIM = StructuralSimilarityIndexMeasure(data_range=1.)



@torch.no_grad()
def ssim_metric(preds, target):
    img1 = Image.open(preds).convert("RGB")
    u = img1.size
    img1 = TF.to_tensor(img1).unsqueeze(0)
    img2 = Image.open(target).convert("RGB")
    img2 = img2.resize(u, Image.Resampling.LANCZOS)
    img2 = TF.to_tensor(img2).unsqueeze(0)

    return SSIM(img1, img2).item()






