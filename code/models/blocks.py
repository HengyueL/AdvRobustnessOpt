import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNetNormalizer(nn.Module):
    def __init__(self, use_clamp_value,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.mean = mean
        self.std = std

        # === if clip input to [0, 1] to ensure valid image ===
        self.use_clamp_value = use_clamp_value

    def forward(self, x):
        if self.use_clamp_value:
            x = torch.clamp(x, 0, 1)
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (
            (x - mean[None, :, None, None]) /
            std[None, :, None, None]
        )


if __name__ == "__main__":
    pass