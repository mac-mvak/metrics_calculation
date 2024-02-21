import torch
import torch.nn.functional as F

from transformers import AutoProcessor, CLIPModel
from PIL import Image



clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

clip_model = clip_model.to(device)
clip_model.eval()


@torch.no_grad()
def image_image_clip_score(img1_path, img2_path):
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)
    image1 = processor(images=image1, return_tensors='pt').to(device)
    image2 = processor(images=image2, return_tensors='pt').to(device)

    features1 = clip_model.get_image_features(**image1)
    features2 = clip_model.get_image_features(**image2)

    ans = F.cosine_similarity(features1, features2)
    return ans.item()

@torch.no_grad()
def image_text_clip_score(img_path, text_prompt):
    image = Image.open(img_path)
    image = processor(images=image, return_tensors='pt').to(device)
    text = processor(text=[text_prompt], padding = True, return_tensors='pt').to(device)
    image_features = clip_model.get_image_features(**image)
    text_features = clip_model.get_text_features(**text)
    ans = F.cosine_similarity(image_features, text_features)

    return ans.item()

