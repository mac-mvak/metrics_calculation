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
        "-r", "--results", type=str, help="Path to directory with results", default='results_imagenet')
    args = parser.parse_args()

    dirs_list = os.listdir(args.results)

    results = defaultdict(list)
    
    for name in tqdm(dirs_list):
        dir_path = args.results + '/' + name

        json_path = dir_path + '/data.json'
        
        with open(json_path) as f:
            json_dict = json.load(f)

        results['time'].append(json_dict['time'])
        results['num_iter'].append(json_dict['num_iter'])

        src_image_path = dir_path + '/input.png'
        for i in range(json_dict['num_iter']):
            gen_image_path = dir_path + f'/output_{i}.png'
            
            results['clip_score'].append(
                image_text_clip_score(gen_image_path, json_dict['prompt_tgt'])
            )
            results['ssim_score'].append(
                ssim_metric(gen_image_path, src_image_path)
            )

avg_time = sum(results['time'])/sum(results['num_iter'])
clip_score = fmean(results['clip_score'])
ssim_score = fmean(results['ssim_score'])
print(f'Average time: {avg_time}')
print(f'Clip_score: {clip_score}')
print(f'SSIM: {ssim_score}')

