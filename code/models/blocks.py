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
        self.clamp_layer = _straightThroughClamp.apply

    def forward(self, x):
        if self.use_clamp_value:
            x = self.clamp_layer(x)
        mean = torch.tensor(self.mean, device=x.device)
        std = torch.tensor(self.std, device=x.device)

        return (
            (x - mean[None, :, None, None]) /
            std[None, :, None, None]
        )


# ==== A straight through [0, 1] clamping ====
class _straightThroughClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return torch.clamp(inputs, 0, 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


if __name__ == "__main__":
    pass