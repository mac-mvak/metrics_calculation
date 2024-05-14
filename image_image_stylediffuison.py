import os
import json
from tqdm import tqdm
import argparse


from statistics import fmean
from collections import defaultdict
from clip_metrics import image_image_clip_score
from photo_metrics import ssim_metric
from style_loss import style_score


if __name__ == '__main__':



    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r", "--results", type=str, help="Path to directory with results", default='runs_gaussian')
    parser.add_argument(
        "-n", '--name', type=str, help="name of the type we want to analyse", default='diffusion')
    parser.add_argument(
        "-t", '--type', type=str, help="name type", default='test')
    args = parser.parse_args()
    images_paths = os.listdir(args.results)
    filtered_images_paths = list(filter(lambda x: x.split('_')[-1] == args.name, images_paths))
    results = defaultdict(list)

    for image_path in tqdm(filtered_images_paths):
        dir_path = os.path.join(args.results, image_path, 'image_samples')
        style_image = os.path.join(dir_path, 'style_color_rec_ninv40.png')
        with open(dir_path + '/time.json') as f:
            time_dict = json.load(f)
        results['time'].append(time_dict['time'])

        name = args.type
        num = 50 if name == 'train' else 5
        r = 6 if name == 'train' else 40
        for i in range(num):
            orig_image_path = os.path.join(dir_path, f'{name}_{i}_0_orig_color.png')
            trans_image_path = os.path.join(dir_path, f'{name}_{i}_2_clip_4_ngen{r}.png')
            results['clip_score'].append(
                image_image_clip_score(trans_image_path, style_image)
                )
            results['ssim_score'].append(
                ssim_metric(orig_image_path, trans_image_path)
                )
                #results['style_score'].append(
                #style_score(trans_image_path, style_image)
                #)

    time_score = fmean(results['time'])
    time_score = time_score / (1000 * 60)
    clip_score = fmean(results['clip_score'])
    ssim_score = fmean(results['ssim_score'])
    print(args.name)
    print(args.type)
    print(f'Time: {time_score}')
    print(f'Clip_score: {clip_score}')
    print(f'SSIM: {ssim_score}')


