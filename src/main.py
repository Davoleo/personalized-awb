import os

from torchvision.transforms import v2
from PIL import Image

from transforms import *

def get_device() -> str:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'  # type: ignore
    print(f"Accelerator: {device}")
    return device

def enhance(transforms, datapath, save_loc):
    """
    Enhance image files in @datapath with @transforms and write them to the @save_loc folders
    """
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(datapath)
        for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    print(paths)
    #sample = random.sample(paths, min(n, len(paths)))

    for path in paths:
        image = Image.open(path).convert('RGB')
        wbalanced = transforms(image)
        newpath = os.path.join(save_loc, os.path.basename(path))
        print(newpath)
        wbalanced.save(newpath)

def main():
    get_device()

    white_patch = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        WhiteBalance(WBAlgorithm.GRAY_WORLD),
        v2.ToPILImage()
    ])
    enhance(white_patch, datapath='../data/Gehler-Shi', save_loc='../data/gray_world_clamped')


if __name__ == "__main__":
    main()
