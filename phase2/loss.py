import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ImageNet normalisation constants — must be applied before every VGG forward pass
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_vgg(img: torch.Tensor) -> torch.Tensor:
    """
    Normalise a batch of images (B, 3, H, W) in [0, 1] to ImageNet stats.
    Works on any device the input is on.
    """
    mean = VGG_MEAN.to(img.device).view(1, 3, 1, 1)
    std  = VGG_STD.to(img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalised Gram matrix of a feature map.

    Args:
        features: (B, C, H, W)
    Returns:
        gram: (B, C, C)  — correlation between every pair of channels,
              normalised by C*H*W so values are scale-independent.
    """
    B, C, H, W = features.shape
    F = features.view(B, C, H * W)           # flatten spatial dims
    G = torch.bmm(F, F.transpose(1, 2))      # (B, C, C) batch matmul
    return G / (C * H * W)


class VGGFeatures(nn.Module):
    """
    Frozen VGG-19 sliced at five layers used for perceptual loss.

    Layer map (VGG-19 .features indices):
        '0'  → relu1_1  (after first conv+relu)
        '5'  → relu2_1
        '10' → relu3_1
        '19' → relu4_1   ← also used as content layer
        '28' → relu5_1

    All parameters are frozen — this module is never updated during training.
    """

    STYLE_LAYERS   = ['0', '5', '10', '19', '28']
    CONTENT_LAYER  = '19'   # relu4_1  (you can try '10' / relu3_1 for softer content)

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)

        # Slice VGG into segments ending at each target layer
        self.slice1 = vgg[:2]    # → relu1_1
        self.slice2 = vgg[2:7]   # → relu2_1
        self.slice3 = vgg[7:12]  # → relu3_1
        self.slice4 = vgg[12:21] # → relu4_1
        self.slice5 = vgg[21:30] # → relu5_1

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns a dict of feature maps keyed by layer name.
        Input must already be VGG-normalised.
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return {'0': h1, '5': h2, '10': h3, '19': h4, '28': h5}


class PerceptualLoss(nn.Module):
    """
    Combined content + style loss using a frozen VGG-19.

    Content loss : MSE between generated and content feature maps at relu4_1.
    Style loss   : MSE between Gram matrices at 5 layers, equally weighted.
    Total loss   : alpha * L_content + beta * L_style

    Usage:
        loss_fn = PerceptualLoss(device, style_image, alpha=1, beta=1e6)
        loss, lc, ls = loss_fn(generated_batch, content_batch)
    """

    def __init__(
        self,
        device:      torch.device,
        style_image: torch.Tensor,   # (1, 3, H, W) in [0, 1]
        alpha:       float = 0.5,
        beta:        float = 1e6,
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.vgg    = VGGFeatures(device)
        self.device = device

        # Pre-compute style Gram matrices once — they never change during training
        with torch.no_grad():
            style_feats = self.vgg(normalize_vgg(style_image.to(device)))
            self.style_grams = {
                layer: gram_matrix(style_feats[layer]).detach()
                for layer in VGGFeatures.STYLE_LAYERS
            }

    def forward(
        self,
        generated: torch.Tensor,   # (B, 3, H, W) — network output
        content:   torch.Tensor,   # (B, 3, H, W) — original content images
    ):
        # Pass both through VGG (normalise first)
        gen_feats     = self.vgg(normalize_vgg(generated))
        content_feats = self.vgg(normalize_vgg(content))

        # --- Content loss ---
        l_content = F.mse_loss(
            gen_feats[VGGFeatures.CONTENT_LAYER],
            content_feats[VGGFeatures.CONTENT_LAYER].detach(),
        )

        # --- Style loss (average over 5 layers) ---
        l_style = 0.0
        w = 1.0 / len(VGGFeatures.STYLE_LAYERS)   # equal layer weights
        for layer in VGGFeatures.STYLE_LAYERS:
            G_gen   = gram_matrix(gen_feats[layer])
            G_style = self.style_grams[layer].expand(G_gen.shape[0], -1, -1)
            l_style = l_style + w * F.mse_loss(G_gen, G_style)

        total = self.alpha * l_content + self.beta * l_style
        return total, l_content.item(), l_style.item()