import os
from .local_dir_dataset import LocalDirDataset

class TrainValDatasetBuilder(object):
    @staticmethod
    def create_train_and_val_data_dir(base_dir, train_path, val_path, train_transform, val_transform):
        full_train_path = os.path.join(base_dir, train_path)
        full_val_path = os.path.join(base_dir, val_path)
        print("train path {}".format(full_train_path))
        print("val path {}".format(full_val_path))

        classes = [os.path.basename(f.path) for f in os.scandir(full_train_path)]
        classes.extend([os.path.basename(f.path) for f in os.scandir(full_val_path)])
        classes = list(set(classes))
        classes.sort()

        return LocalDirDataset.create_with_class_dir(full_train_path, classes, train_transform), LocalDirDataset.create_with_class_dir(full_val_path, classes, val_transform)

    @staticmethod
    def create_train_and_val_data_with_index_file(base_dir, train_path, val_path, class_file, train_transform, val_transform):
        print("train index file path {}".format(train_path))
        print("val indexfile path {}".format(val_path))

        classes = None
        with open(os.path.join(base_dir, class_file), 'r') as classes_in:
            classes = [line.strip() for line in classes_in]

        return LocalDirDataset.create_with_index_file(base_dir, classes, train_path, train_transform), LocalDirDataset.create_with_index_file(base_dir, classes, val_path, train_transform)




