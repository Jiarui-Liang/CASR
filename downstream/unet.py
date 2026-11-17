import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import vision_transformer as vits


class UNet(nn.Module):
    def __init__(self, num_classes, encoder_name, patch_size=8):
        super().__init__()

        self.encoder = vits.__dict__[encoder_name](patch_size=patch_size, num_classes=0, apply_head=False)


        self.encoder_channels = [768, 768, 768, 768]

        self.decoder_channels = [512, 256, 128, 64]

        self.center = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], self.decoder_channels[0], kernel_size=1),
            nn.BatchNorm2d(self.decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.up1 = self._up_block(self.decoder_channels[0], self.encoder_channels[2], self.decoder_channels[1])
        self.up2 = self._up_block(self.decoder_channels[1], self.encoder_channels[1], self.decoder_channels[2])
        self.up3 = self._up_block(self.decoder_channels[2], self.encoder_channels[0], self.decoder_channels[3])

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def _extract_spatial_features(self, x):
        enc_features = self.encoder(x)
        if isinstance(enc_features, (list, tuple)):
            features = list(enc_features)
        else:
            features = [enc_features]

        spatial_features = []
        for feat in features:
            if feat.dim() == 3:
                token_count = feat.shape[1]
                side = int(math.sqrt(token_count))
                if side * side != token_count and token_count > 1:
                    feat = feat[:, 1:, :]
                    token_count = feat.shape[1]
                    side = int(math.sqrt(token_count))
                feat = feat.permute(0, 2, 1)
                if side * side != token_count:
                    raise ValueError(f"Token count {token_count} is not a square.")
                feat = feat.view(feat.shape[0], feat.shape[1], side, side)
            elif feat.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")
            spatial_features.append(feat)

        if len(spatial_features) < 4:
            raise ValueError(f"Need at least 4 feature maps, got {len(spatial_features)}.")
        return spatial_features[-4:]

    def _up_block(self, in_channels, skip_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        spatial_features = self._extract_spatial_features(x)

        # Center
        center = self.center(spatial_features[-1])  # [B, 512, 28, 28]

        # Decoder with skip connections
        x = self.up1[0](center)  # [B, 512, 56, 56]
        spatial_features[2] = F.interpolate(spatial_features[2], size=(x.shape[-1], x.shape[-1]), mode='bilinear', align_corners=True)
        x = torch.cat([x, spatial_features[2]], dim=1)  # [B, 512+768=1280, 56, 56]
        x = self.up1[1:](x)  # [B, 256, 56, 56]

        x = self.up2[0](x)  # [B, 256, 112, 112]
        spatial_features[1] = F.interpolate(spatial_features[1], size=(x.shape[-1], x.shape[-1]), mode='bilinear', align_corners=True)
        x = torch.cat([x, spatial_features[1]], dim=1)  # [B, 256+768=1024, 112, 112]
        x = self.up2[1:](x)  # [B, 128, 112, 112]

        x = self.up3[0](x)  # [B, 128, 224, 224]
        spatial_features[0] = F.interpolate(spatial_features[0], size=(x.shape[-1], x.shape[-1]), mode='bilinear', align_corners=True)
        x = torch.cat([x, spatial_features[0]], dim=1)  # [B, 128+768=896, 224, 224]
        x = self.up3[1:](x)  # [B, 64, 224, 224]

        # Final output
        x = self.final_conv(x)  # [B, num_classes, 224, 224]
        return x
