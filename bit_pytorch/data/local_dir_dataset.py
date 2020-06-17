import torch
import os
from PIL import Image
from sklearn.utils import shuffle

class LocalDirDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, classes, transform):
        super().__init__()
        self.transform = transform

        self.image_paths = []
        self.image_targets = []

        self.classes = classes
        print(self.classes)

        c_idx = -1
        for c in self.classes:
            c_idx = c_idx + 1
            c_path = os.path.join(basedir, c)
            print(c_path)
            if not os.path.isdir(c_path):
                continue
            img_paths = [f.path for f in os.scandir(c_path)]
            self.image_paths.extend(img_paths)
            self.image_targets.extend([c_idx] * len(img_paths))
        
        assert len(self.image_paths) == len(self.image_targets)
        
        self.image_paths, self.image_targets = shuffle(self.image_paths, self.image_targets)

    def __getitem__(self, index):
        image = None
        idx = index-1
        while image == None and idx - index < 10:
            idx = (idx + 1) % self.__len__()
            image = self._load_image(self.image_paths[idx])

        if idx - index == 10:
            raise Exception("Something wrong about the dataset.")

        image = self.transform(image)

        return image, self.image_targets[idx]

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                image = Image.open(f)
                return image.convert('RGB')
        except Exception:
            print(f"Failed to load an image: {image_path}")
            return None