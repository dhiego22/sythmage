def Norm3d(num_channels, norm_type="instance", num_groups=8):
    if norm_type == "group":
        # Ensure num_groups divides num_channels; fallback to 4 or 1 if needed
        g = num_groups
        if num_channels % g != 0:
            g = 4 if num_channels % 4 == 0 else 1
        return nn.GroupNorm(g, num_channels)
    else:
        return nn.InstanceNorm3d(num_channels, affine=True)

class DownBlock3D(nn.Module):  # ENCODER
    def __init__(self, in_ch, out_ch, norm=True, norm_type="instance"):
        super().__init__()
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
        if norm:
            layers.append(Norm3d(out_ch, norm_type=norm_type))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class UpBlock3D(nn.Module):  # DECODER
    def __init__(self, in_ch, out_ch, dropout=False, norm_type="instance", up_mode="trilinear", blur=True):
        super().__init__()
        self.up_mode = up_mode  # "nearest" or "trilinear"
        self.blur = blur
        if blur:
            self.blur_conv = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1,
                                       groups=in_ch, bias=False)
            with torch.no_grad():
                k = torch.tensor([1., 2., 1.])
                k3 = (k[:,None,None]*k[None,:,None]*k[None,None,:])
                k3 = k3 / k3.sum()
                w = torch.zeros((in_ch,1,3,3,3))
                w[:,0] = k3
                self.blur_conv.weight.copy_(w)
            for p in self.blur_conv.parameters():
                p.requires_grad = False
        else:
            self.blur_conv = nn.Identity()

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = Norm3d(out_ch, norm_type=norm_type)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(0.5) if dropout else nn.Identity()

    def forward(self, x):
        if self.up_mode == "trilinear":
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode)  # "nearest" recommended for anti-checkerboard
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class FinalUp3D(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode="nearest", out_activation="tanh"):
        super().__init__()
        self.up_mode = up_mode
        self.out_conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        if out_activation == "tanh":
            self.out_act = nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

    def forward(self, x):
        if self.up_mode == "trilinear":
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class UNetGenerator3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, norm_type="instance", up_mode="nearest", out_activation="tanh"):
        super().__init__()

        # Encoder
        self.d1 = DownBlock3D(in_ch, 64,  norm=False, norm_type=norm_type)  # no norm on first layer
        self.d2 = DownBlock3D(64,  128, norm=True,  norm_type=norm_type)
        self.d3 = DownBlock3D(128, 256, norm=True,  norm_type=norm_type)
        self.d4 = DownBlock3D(256, 512, norm=True,  norm_type=norm_type)

        # Bottleneck (BN -> IN/GN)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1, bias=False),
            Norm3d(512, norm_type=norm_type),
            nn.ReLU(inplace=True),
        )

        # Decoder (ConvTranspose3d -> interpolate+conv)
        self.u1 = UpBlock3D(512, 256, dropout=False,  norm_type=norm_type, up_mode=up_mode)  # b -> 256
        self.u2 = UpBlock3D(512, 128, dropout=False,  norm_type=norm_type, up_mode=up_mode)  # cat(u1,d3)=256+256=512 -> 128
        self.u3 = UpBlock3D(256, 64,  dropout=False, norm_type=norm_type, up_mode=up_mode)  # cat(u2,d2)=128+128=256 -> 64

        # Final (cat(u3,d1)=64+64=128 -> out)
        self.u4 = FinalUp3D(128, out_ch, up_mode=up_mode, out_activation=out_activation)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        b = self.bottleneck(d4)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d3], dim=1))
        u3 = self.u3(torch.cat([u2, d2], dim=1))
        out = self.u4(torch.cat([u3, d1], dim=1))
        return out


class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_ch=2, norm_type="instance"):
        super().__init__()

        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(Norm3d(out_ch, norm_type=norm_type))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            block(64, 128, norm=True),
            block(128, 256, norm=True),
            block(256, 512, norm=True),
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        return self.model(x)
