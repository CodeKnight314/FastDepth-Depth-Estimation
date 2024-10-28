import matplotlib.pyplot as plt
import argparse
from PIL import Image

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    
    args = parser.parse_args()
    
    plt.imshow(Image.open(args.path))
    plt.show()