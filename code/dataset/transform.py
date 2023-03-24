import torch as ch
from torchvision import transforms


IMAGENET_PCA = {
    'eigval':ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])


TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        # transforms.ToTensor()
    ])


