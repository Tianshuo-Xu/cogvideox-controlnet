import os, io, csv, math, random
import numpy as np
from einops import rearrange
import pandas as pd

import torch
from decord import VideoReader, cpu
import torch.distributed as dist

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import h5py
import cv2
from copy import deepcopy


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class YoutubeVideoData(Dataset):
    def __init__(
            self,
            meta_path='/hpc2hdd/home/txu647/code/video_data/metas_40v_driving/meta_clips_caption_cleaned.csv',
            data_dir='/hpc2hdd/home/txu647/code/video_data/clips',
            sample_size=[256, 256], 
            sample_stride=[1, 4], 
            sample_n_frames=14,
            random_flip=True,
            random_crop=True,
        ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['text']
        del metadata['text']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir
        self.random_flip = random_flip
        self.random_crop = random_crop

        print(f"random flip: {random_flip}, random crop: {random_crop}")
        self.length = len(self.metadata)
        print(f"data scale: {self.length}")

        self.sample_stride = sample_stride
        print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.sample_size = sample_size
        self.temp_sample_size = deepcopy(self.sample_size)
        if self.random_crop:
            self.temp_sample_size[0] = self.sample_size[0] + self.sample_size[0] // 8
            self.temp_sample_size[1] = self.sample_size[1] + self.sample_size[1] // 8
    
    def _get_video_path(self, sample):
        rel_video_fp = sample['path'].split('/')[-1]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp
    
    def get_batch(self, index):
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path, rel_path = self._get_video_path(sample)

            sample_stride = random.randint(self.sample_stride[0], self.sample_stride[1])
            required_frame_num = sample_stride * self.sample_n_frames

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < required_frame_num:
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            frame_num = len(video_reader)

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + sample_stride*i for i in range(self.sample_n_frames)]

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        assert(frames.shape[0] == self.sample_n_frames),f'{len(frames)}, self.video_length={self.sample_n_frames}'

        frames = frames.asnumpy()

        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(Image.fromarray(frames[i]).convert('RGB').resize(
                [self.temp_sample_size[1], self.temp_sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]
    
        return resized_frames, rel_path
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, video_name = self.get_batch(idx)
        if self.random_crop:
            crop_y = random.randrange(pixel_values.shape[2] - self.sample_size[0] + 1)
            crop_x = random.randrange(pixel_values.shape[3] - self.sample_size[1] + 1)
            pixel_values = pixel_values[:, :, crop_y : crop_y + self.sample_size[0], crop_x : crop_x + self.sample_size[1]]
            video_name += '_crop'

        if self.random_flip and random.random() <= 0.5:
            pixel_values = torch.flip(pixel_values, dims=[3])
            video_name += '_flip'

        pixel_values = pixel_values / 127.5 - 1
        
        sample = dict(video=pixel_values, caption="a realistic driving scenario with high visual quality, high resolution")
        return sample


if __name__ == '__main__':
    breakpoint()
