import matplotlib.pyplot as plt
import numpy as np 
import argparse 
from PIL import Image

def show_img(path: str):
    img = Image.open(path).convert("RGB")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to image display")
    
    args = parser.parse_args()
    
    show_img(args.path)
    
    