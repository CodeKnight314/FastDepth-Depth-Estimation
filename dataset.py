from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch
import argparse

class NYU_DepthDataset(Dataset):
    def __init__(self, input_dir: str, mode: str):
        super().__init__()
        
        self.rgb_images = sorted(glob(os.path.join(input_dir, mode, "RGB" "/*")))
        self.depth_images = sorted(glob(os.path.join(input_dir, mode, "Depth" "/*")))
        
        assert len(self.rgb_images) == len(self.depth_images), "Number of RGB and depth images should be the same."
        
        self.rgb_transform = T.ToTensor()
        self.depth_transform = T.ToTensor()
        
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, index: int):
        rgb_img = Image.open(self.rgb_images[index]).convert("RGB")
        rgb_tensor = self.rgb_transform(rgb_img)
        
        depth_img = Image.open(self.depth_images[index])
        depth_np = np.array(depth_img, dtype=np.float32)
        
        max_depth_value = depth_np.max() if depth_np.max() > 0 else 1.0
        depth_np = depth_np / max_depth_value
        
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        
        return rgb_tensor, depth_tensor
    
def load_dataset(input_dir: str, mode: str, batch_size: int):
    dataset = NYU_DepthDataset(input_dir, mode)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count())
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path for loading NYU Depth dataset")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for dataloading")

    args = parser.parse_args()
    
    train_loader = load_dataset(args.path, "train", args.batch_size)
    test_loader = load_dataset(args.path, "test", args.batch_size)
    
    tr_rgb, tr_depth = next(iter(train_loader))
    ts_rgb, ts_depth = next(iter(test_loader))
    
    def inspect_data(rgb_tensor, depth_tensor):
        print("RGB Tensor Shape:", rgb_tensor.shape)
        print("Depth Tensor Shape:", depth_tensor.shape)
        
        print("RGB Tensor Range:", rgb_tensor.min().item(), "-", rgb_tensor.max().item())
        print("Depth Tensor Range:", depth_tensor.min().item(), "-", depth_tensor.max().item())

    inspect_data(tr_rgb, tr_depth)
    inspect_data(ts_rgb, ts_depth)
