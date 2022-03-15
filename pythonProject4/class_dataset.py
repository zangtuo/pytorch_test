from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.datasets


class MyData(Dataset):

    def __init__(self, root_dir, label_dir, transform=torchvision.transforms.ToTensor):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.label_dir
        return img, lable

    def __len__(self):
        return len(self.img_path)


data = MyData("data_ants_bees/train", "ants")
print(data[1])
