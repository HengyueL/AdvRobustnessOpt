import torch
from robustness import loaders
import robustness.data_augmentation as da


# ==== Below realizes the ImageNet-100 dataset ====

def get_label_mapping(dataset_name, ranges):
    if dataset_name == 'imagenet':
        label_mapping = None
    elif dataset_name == 'restricted_imagenet':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    elif dataset_name == 'custom_imagenet':
        def label_mapping(classes, class_to_idx):
            return custom_label_mapping(classes, class_to_idx, ranges=ranges)
    else:
        raise ValueError('No such dataset_name %s' % dataset_name)

    return label_mapping


def restricted_label_mapping(classes, class_to_idx, ranges):
    range_sets = [
        set(range(s, e+1)) for s,e in ranges
    ]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping


def custom_label_mapping(classes, class_to_idx, ranges):

    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping[class_name] = new_idx

    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std', 
                         'transform_train', 'transform_test']
        optional_args = ['custom_class', 'label_mapping', 'custom_class_args']

        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args} 

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)
    
    def override_args(self, default_args, kwargs):
        '''
        Convenience method for overriding arguments. (Internal)
        '''
        for k in kwargs:
            if not (k in default_args): continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}


    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, 
                    subset_start=0, subset_type='rand', val_batch_size=None,
                    only_val=False, shuffle_train=True, shuffle_val=True, subset_seed=None):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128) 
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val,
                                    seed=subset_seed,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val,
                                    custom_class_args=self.custom_class_args)


class CustomImageNet(DataSet):
    '''
    CustomImagenet Dataset 

    A subset of ImageNet with the user-specified labels

    To initialize, just provide the path to the full ImageNet dataset
    along with a list of lists of wnids to be grouped together
    (no special formatting required).

    '''
    def __init__(
        self, data_path, 
        custom_grouping, 
        train_transform=None,
        **kwargs):
        assert train_transform is not None, "Train trainsform cannot be none."

        ds_name = 'custom_imagenet'
        ds_kwargs = {
            'num_classes': len(custom_grouping),
            'mean': torch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': torch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': get_label_mapping(ds_name,
                custom_grouping),
            'transform_train': train_transform,
            'transform_test': da.TEST_TRANSFORMS_IMAGENET
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(CustomImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)


class ImageNetN(CustomImageNet):
    def __init__(
        self, data_path, 
        label_list=None,
        train_transform=None,
        **kwargs):
        assert label_list is not None, "Need To specify the label class."
        assert train_transform is not None, "Need to specify the train transform."
        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in label_list],
            train_transform=train_transform,
            **kwargs,
        )


if __name__ ==  "__main__":
    print()

