import os
import json
from tqdm import tqdm
import argparse


from statistics import fmean
from collections import defaultdict
from clip_metrics import image_text_clip_score
from photo_metrics import ssim_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r", "--results", type=str, help="Path to directory with results", default='results_default')
    parser.add_argument(
        "-i", '--images', type=str, help="Path to directory with initial_images", default='imagenet_subset')
    parser.add_argument(
        "-j", "--prompt_json", type=str, help='JSON with text prompts', default='texts.json')
    parser.add_argument(
        "-k", "--num_samples", type=int, default=2)
    args = parser.parse_args()
    with open('texts.json') as f:
        prompts = json.load(f)
    prompts.sort()
    images_paths = os.listdir(args.images)
    images_paths.sort()
    results = defaultdict(list)

    for prompt in tqdm(prompts):
        for i, image_path in enumerate(images_paths):
            src_image_path = os.path.join(args.images, image_path)
            for k in range(args.num_samples):
                gen_image_path = os.path.join(args.results, prompt, f"{i}", f"output_{k}.png")
                results['clip_score'].append(
                image_text_clip_score(gen_image_path, prompt)
                )
                results['ssim_score'].append(
                ssim_metric(gen_image_path, src_image_path)
                )
                

    clip_score = fmean(results['clip_score'])
    ssim_score = fmean(results['ssim_score'])
    print(args.results)
    print(f'Clip_score: {clip_score}')
    print(f'SSIM: {ssim_score}')


