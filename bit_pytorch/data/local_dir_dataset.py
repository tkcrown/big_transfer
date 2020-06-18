import torch
import os
from PIL import Image
from sklearn.utils import shuffle

class LocalDirDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_targets, classes, transform):
        self.image_paths = image_paths
        self.image_targets = image_targets
        self.classes = self.classes
        self.transform = transform

    @staticmethod
    def create_with_class_dir(basedir, classes, transform):
        super().__init__()

        image_paths = []
        image_targets = []

        print(classes)

        c_idx = -1
        for c in classes:
            c_idx = c_idx + 1
            c_path = os.path.join(basedir, c)
            print(c_path)
            if not os.path.isdir(c_path):
                continue
            img_paths = [f.path for f in os.scandir(c_path)]
            image_paths.extend(img_paths)
            image_targets.extend([c_idx] * len(img_paths))
        
        assert len(image_paths) == len(image_targets)
        
        image_paths, image_targets = shuffle(image_paths, image_targets)
        return LocalDirDataset(image_paths, image_targets, classes, transform)

    @staticmethod
    def create_with_index_file(basedir, classes, index_file_name, transform):
        super().__init__()

        image_paths = []
        image_targets = []

        print(classes)

        class_to_idx = {}
        for i,c in enumerate(classes):
            class_to_idx[c] = i

        with open(os.path.join(basedir, index_file_name), 'r') as file_in:
            for line in file_in:
                path, c = line.strip().split('\t')
                image_paths.append(os.path.join(basedir, path))
                image_targets.append(class_to_idx[c])

        image_paths, image_targets = shuffle(image_paths, image_targets)
        return LocalDirDataset(image_paths, image_targets, classes, transform)

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