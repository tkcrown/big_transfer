import os
from .local_dir_dataset import LocalDirDataset

class TrainValDatasetBuilder(object):
    @staticmethod
    def create_train_and_val_data(base_dir, train_path, val_path, train_transform, val_transform):
        full_train_path = os.path.join(base_dir, train_path)
        full_val_path = os.path.join(base_dir, val_path)
        print("train path {}".format(full_train_path))
        print("val path {}".format(full_val_path))
        print()
        classes = [os.path.basename(f.path) for f in os.scandir(full_train_path)]
        classes.extend([os.path.basename(f.path) for f in os.scandir(full_val_path)])
        classes = list(set(classes))
        classes.sort()

        return LocalDirDataset(full_train_path, classes, train_transform), LocalDirDataset(full_val_path, classes, val_transform)




