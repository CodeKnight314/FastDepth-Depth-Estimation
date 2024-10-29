from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch

class NYU_DepthDataset(Dataset):
    def __init__(self, input_dir: str, mode: str):
        super().__init__()
        
        self.rgb_images = sorted(glob(os.path.join(input_dir, mode, "RGB" "*")))
        self.depth_images = sorted(glob(os.path.join(input_dir, mode, "Depth" "*")))
        
        assert len(self.rgb_images) == len(self.depth_images), "Number of RGB and depth images should be the same."
        
        self.rgb_transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.depth_transform = T.ToTensor()
        
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, index):
        rgb_img = Image.open(self.rgb_images[index]).convert("RGB")
        rgb_tensor = self.rgb_transform(rgb_img)
        
        depth_img = Image.open(self.depth_images[index])
        depth_np = np.array(depth_img, dtype=np.float32)
        
        max_depth_value = 65535.0 if depth_np.dtype == np.uint16 else 255.0
        depth_np = depth_np / max_depth_value
        
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        
        return rgb_tensor, depth_tensor
    
def load_dataset(input_dir: str, mode: str, batch_size: int):
    dataset = NYU_DepthDataset(input_dir, mode)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count())
    return dataloader