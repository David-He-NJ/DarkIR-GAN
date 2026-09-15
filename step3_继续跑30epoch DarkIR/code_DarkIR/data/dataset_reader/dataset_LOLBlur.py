import os
import random
from glob import glob

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import torch

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *


class SceneBasedSampler(Sampler):
    """
    基于场景的采样器，每个epoch从每个场景中随机选择固定数量的图片
    """
    def __init__(self, scene_paths_blur, scene_paths_sharp, num_samples=15, shuffle=True):
        self.scene_paths_blur = sorted(scene_paths_blur)
        self.scene_paths_sharp = sorted(scene_paths_sharp)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.epoch = 0
        
        # 执行初始采样
        self._resample()
    
    def _get_all_images_from_scenes(self, scene_paths):
        """获取所有场景中的所有图片，按场景组织"""
        all_scenes_images = []
        for scene_path in scene_paths:
            images = [os.path.join(scene_path, img) for img in os.listdir(scene_path)
                      if img.endswith(('.png', '.jpg', '.jpeg'))]
            images.sort()
            all_scenes_images.append(images)
        return all_scenes_images
    
    def _resample(self):
        """每个场景重新采样图片"""
        blur_scenes = self._get_all_images_from_scenes(self.scene_paths_blur)
        sharp_scenes = self._get_all_images_from_scenes(self.scene_paths_sharp)
        
        selected_blur = []
        selected_sharp = []
        
        for blur_images, sharp_images in zip(blur_scenes, sharp_scenes):
            # 确保模糊和清晰图片数量一致
            assert len(blur_images) == len(sharp_images), f"场景 {blur_images[0] if blur_images else 'unknown'} 中模糊和清晰图片数量不匹配"
            
            n_images = len(blur_images)
            if self.num_samples <= 0 or n_images <= self.num_samples:
                # 如果num_samples <= 0（使用所有图片）或图片数量少于num_samples，则取全部
                selected_indices = list(range(n_images))
            else:
                # 随机选择 num_samples 张
                selected_indices = random.sample(range(n_images), self.num_samples)
            
            for idx in selected_indices:
                selected_blur.append(blur_images[idx])
                selected_sharp.append(sharp_images[idx])
        
        # 创建索引列表
        self.indices = list(range(len(selected_blur)))
        self.current_blur = selected_blur
        self.current_sharp = selected_sharp
        
        if self.shuffle:
            random.shuffle(self.indices)
        
        return selected_blur, selected_sharp
    
    def set_epoch(self, epoch):
        """在每个 epoch 开始时调用，重新采样"""
        self.epoch = epoch
        self._resample()
    
    def get_current_paths(self):
        """获取当前采样的图片路径"""
        return self.current_blur, self.current_sharp
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


def main_dataset_lolblur(rank = 1,
                         test_path='datasets/LOLBlur/train',
                         batch_size_test=1,
                         verbose=False,
                         num_workers=1,
                         world_size = 1,
                         return_path=False,
                         samples_per_scene=15):
    
    PATH_VALID = test_path
    
    # paths to the blur and sharp sets of images
    paths_blur_scenes = [os.path.join(PATH_VALID, 'low_blur', path) for path in os.listdir(os.path.join(PATH_VALID, 'low_blur'))]
    paths_sharp_scenes = [os.path.join(PATH_VALID, 'high_sharp_original', path) for path in os.listdir(os.path.join(PATH_VALID, 'high_sharp_original'))]
    
    # 创建场景采样器
    scene_sampler = SceneBasedSampler(
        paths_blur_scenes,
        paths_sharp_scenes,
        num_samples=samples_per_scene,
        shuffle=True
    )
    
    # 获取初始采样的图片路径
    list_blur_valid, list_sharp_valid = scene_sampler.get_current_paths()
    
    # 确保模糊和清晰图片数量一致
    assert len(list_blur_valid) == len(list_sharp_valid), "模糊和清晰图片数量不匹配"

    # check if all the image routes are correct
    check_paths([list_blur_valid, list_sharp_valid])

    if verbose:
        print('Images in the subsets:')
        print("    -Images in the PATH_LOW_VALID folder: ", len(list_blur_valid))
        print("    -Images in the PATH_HIGH_VALID folder: ", len(list_sharp_valid))
        print(f"    -Number of scenes: {len(paths_blur_scenes)}")
        print(f"    -Samples per scene: {samples_per_scene}")

    tensor_transform = transforms.ToTensor()

    # Load the dataset
    test_dataset = MyDataset_Crop(list_blur_valid, list_sharp_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True, return_path=return_path)
    
    if world_size > 1:
        # Now we need to apply the Distributed sampler
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle= True, rank=rank)

        samplers = []
        # samplers = {'train': train_sampler, 'test': [test_sampler_gopro, test_sampler_lolblur]}
        samplers.append(test_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:
        # 使用场景采样器
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                sampler=scene_sampler)
        samplers = scene_sampler

    return test_loader, samplers

if __name__ == '__main__':
    test_loader, samplers = main_dataset_lolblur()
    
