import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utils / Norm
# =========================
def norm2d(num_ch, kind="bn"):
    if kind == "bn":
        return nn.BatchNorm2d(num_ch)
    elif kind == "gn":
        groups = min(16, num_ch)
        while num_ch % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_ch)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")

def norm3d(num_ch, kind="bn"):
    if kind == "bn":
        return nn.BatchNorm3d(num_ch)
    elif kind == "gn":
        groups = min(16, num_ch)
        while num_ch % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_ch)
    else:
        raise ValueError(f"Unknown norm kind: {kind}")


# =========================
# Blocks
# =========================
class ResidualBlock(nn.Module):
    """Standard ResNet Block used in Encoder and Decoder"""
    def __init__(self, in_ch, out_ch, dropout=0.0, norm="bn"):
        super().__init__()
        # always match channels
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = norm2d(out_ch, norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = norm2d(out_ch, norm)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        if self.proj is not None:
            identity = self.proj(x)
        h = self.relu(self.n1(self.conv1(x)))
        h = self.drop(self.n2(self.conv2(h)))
        return self.relu(h + identity)

class FusionBlock(nn.Module):
    """
    Implements the Gated Fusion Block (GFB) described in Eq. 3.
    f_TE = f_T + alpha * f_E
    """
    def __init__(self, ch_t, ch_e, mode='gated', init_lambda=0.4):
        super().__init__()
        assert mode in ('gated', 'add', 'concat'), f"Unknown fusion mode: {mode}"
        self.mode = mode

        # Project events into channel space of thermal
        self.pe = nn.Conv2d(ch_e, ch_t, kernel_size=1, bias=True)
        # start conservative: events initially off
        nn.init.zeros_(self.pe.weight)
        nn.init.zeros_(self.pe.bias)

        # Gated params (Eq. 3)
        if self.mode == 'gated':
            self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda)))
        else:
            self.log_lambda = None 

        # Concat reduce conv
        if self.mode == 'concat':
            self.reduce = nn.Conv2d(2 * ch_t, ch_t, kernel_size=1, bias=True)
            with torch.no_grad():
                self.reduce.weight.zero_()
                self.reduce.bias.zero_()
                eye = torch.eye(ch_t)
                self.reduce.weight[:, :ch_t, 0, 0] = eye
        else:
            self.reduce = None

        self.alpha_mean = None 

    def forward(self, t, e):
        # Align spatial dimensions if needed
        if e.shape[2:] != t.shape[2:]:
            e = F.interpolate(e, size=t.shape[2:], mode='bilinear', align_corners=False)
        
        e = self.pe(e) # Project events to match thermal channels

        if self.mode == 'gated':
            lam = torch.exp(self.log_lambda) # scalar alpha > 0
            out = t + lam * e
            
            # Log metric (optional)
            alpha_eff = 1.0 / (1.0 + lam)
            self.alpha_mean = torch.as_tensor(alpha_eff, device=t.device)
            return out

        elif self.mode == 'add':
            out = t + e
            return out

        else: # 'concat'
            out = self.reduce(torch.cat([t, e], dim=1))
            return out


# =========================
# Encoders
# =========================
class Encoder(nn.Module):
    """Thermal Encoder: 4-stage residual pyramid"""
    def __init__(self, in_ch, base=32, dropout=0.0, norm="bn"):
        super().__init__()
        C = base
        # Hierarchy: 32 -> 64 -> 128 -> 256 [cite: 363]
        self.b1 = ResidualBlock(in_ch, C, dropout, norm)
        self.p1 = nn.MaxPool2d(2)

        self.b2 = ResidualBlock(C, C*2, dropout, norm)
        self.p2 = nn.MaxPool2d(2)

        self.b3 = ResidualBlock(C*2, C*4, dropout, norm)
        self.p3 = nn.MaxPool2d(2)

        self.b4 = ResidualBlock(C*4, C*8, dropout, norm)

    def forward(self, x):
        f1 = self.b1(x)             # [B,32,H,W]
        f2 = self.b2(self.p1(f1))   # [B,64,H/2,W/2]
        f3 = self.b3(self.p2(f2))   # [B,128,H/4,W/4]
        f4 = self.b4(self.p3(f3))   # [B,256,H/8,W/8]
        return f1, f2, f3, f4

class EventEncoder2D(nn.Module):
    def __init__(self, in_bins=5, base=32, dropout=0.0, norm="bn"):
        super().__init__()
        self.enc = Encoder(in_bins, base=base, dropout=dropout, norm=norm)

    def forward(self, ev):
        if ev.dim() == 5:   # [B,T,1,H,W] -> [B,T,H,W]
            ev = ev.squeeze(2)
        return self.enc(ev)

class EventEncoder3D(nn.Module):
    """
    3D Conv Head + 2D Encoder
    """
    def __init__(self, base=32, dropout=0.0, norm="bn"):
        super().__init__()
        # "lightweight 3D convolutional block... two 3x3x3 Conv3D" [cite: 366]
        self.stem = nn.Sequential(
            nn.Conv3d(1, 8, (3,3,3), padding=1, bias=False),
            norm3d(8, norm),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, (3,3,3), padding=1, bias=False),
            norm3d(16, norm),
            nn.ReLU(inplace=True),
        )
        # "produce 16 feature maps... collapsed over time" [cite: 366]
        self.proj = nn.Sequential(
            nn.Conv2d(16, base, 3, padding=1, bias=False),
            norm2d(base, norm),
            nn.ReLU(inplace=True),
        )
        # Mirrors thermal encoder structure [cite: 370]
        self.enc2 = ResidualBlock(base, base*2, dropout, norm)
        self.enc3 = ResidualBlock(base*2, base*4, dropout, norm)
        self.enc4 = ResidualBlock(base*4, base*8, dropout, norm)

    def forward(self, ev):
        if ev.dim() == 4:               # [B,T,H,W]
            x = ev.unsqueeze(1)         # [B,1,T,H,W]
        elif ev.dim() == 5 and ev.shape[2] == 1:
            x = ev.permute(0, 2, 1, 3, 4) # [B,1,T,H,W]
        else:
            raise ValueError(f"Unexpected events shape: {ev.shape}")
        
        h = self.stem(x).mean(dim=2)    # Temporal Mean [cite: 366]
        f1 = self.proj(h)               # [B,32,H,W]
        
        # MaxPool is applied before the block in the standard flow to match sizes
        f2 = self.enc2(F.max_pool2d(f1, 2))  # [B,64,H/2,W/2]
        f3 = self.enc3(F.max_pool2d(f2, 2))  # [B,128,H/4,W/4]
        f4 = self.enc4(F.max_pool2d(f3, 2))  # [B,256,H/8,W/8]
        return f1, f2, f3, f4

# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    """
    Shared decoder with U-Net style skip connections.
    Takes fused features f1..f4.
    """
    def __init__(self, base=32, dropout=0.0, norm="bn"):
        super().__init__()
        C = base

        # Stage 3: Upsample -> Concat f3 -> ResBlock [cite: 379]
        self.up3 = nn.ConvTranspose2d(C*8, C*4, 2, stride=2)
        self.d3  = ResidualBlock(C*8, C*4, dropout, norm) # in: (C*4 from up + C*4 from skip) = C*8

        # Stage 2
        self.up2 = nn.ConvTranspose2d(C*4, C*2, 2, stride=2)
        self.d2  = ResidualBlock(C*4, C*2, dropout, norm)

        # Stage 1
        self.up1 = nn.ConvTranspose2d(C*2, C, 2, stride=2)
        self.d1  = ResidualBlock(C*2, C, dropout, norm)

        # Final projection to Lab [cite: 381]
        self.final = nn.Conv2d(C, 3, 1)

    def forward(self, f1, f2, f3, f4):
        # f4 is the bottleneck
        x = self.up3(f4)
        x = torch.cat([x, f3], dim=1) # Skip connection
        d3 = self.d3(x)

        x = self.up2(d3)
        x = torch.cat([x, f2], dim=1)
        d2 = self.d2(x)

        x = self.up1(d2)
        x = torch.cat([x, f1], dim=1)
        d1 = self.d1(x)

        raw = self.final(d1)
        # "Sigmoid (L) ... Tanh (ab)" [cite: 381]
        L  = torch.sigmoid(raw[:, :1])   # [0,1]
        ab = torch.tanh(raw[:, 1:])      # [-1,1]
        return torch.cat([L, ab], dim=1)

# =========================
# Full Network
# =========================
class ThermalEvent2RGBNet(nn.Module):
    """
    Main Framework described in Fig. 1.
    """
    def __init__(self,
                 event_bins=5,
                 dropout=0.3,
                 encoder_type='3d',     # Default to '3d' as per paper results
                 base=32,
                 norm='bn',
                 fusion='gated'         # Default to 'gated' 
                 ):
        super().__init__()
        assert encoder_type in ['2d', '3d']
        assert fusion in ['gated', 'add', 'concat']
        self.encoder_type = encoder_type

        # Encoders
        self.enc_th = Encoder(1, base=base, dropout=dropout, norm=norm)
        
        if encoder_type == '2d':
            self.enc_ev = EventEncoder2D(event_bins, base=base, dropout=dropout, norm=norm)
        else:
            self.enc_ev = EventEncoder3D(base=base, dropout=dropout, norm=norm)

        # Fusion Module 
        self.f1 = FusionBlock(base,     base,     mode=fusion)
        self.f2 = FusionBlock(base*2,   base*2,   mode=fusion)
        self.f3 = FusionBlock(base*4,   base*4,   mode=fusion)
        self.f4 = FusionBlock(base*8,   base*8,   mode=fusion)

        # Decoder 
        self.dec = Decoder(base=base, dropout=dropout, norm=norm)

    def _events_to_pyramid(self, events):
        """Helper to forward events through correct encoder"""
        if events.dim() == 5: 
            events = events.squeeze(2) # ensure [B,T,H,W] for 2D logic or keep for 3D inside encoder check
            # Note: EventEncoder3D handles dimensions internally, 
            # but if we passed 5D to squeeze above, we need to unsqueeze inside 3D encoder or fix here.
            # To be safe, let's restore the squeeze only for 2D.
            if self.encoder_type == '3d':
                events = events.unsqueeze(2) # restore [B,T,1,H,W] for 3D

        return self.enc_ev(events)

    def forward(self, thermal, events):
        """
        Standard forward pass fusing Thermal + Events.
        Handles missing modalities by zeroing features if needed.
        """
        have_th = thermal is not None
        have_ev = events  is not None

        if not have_th and not have_ev:
            raise ValueError("At least one input required.")

        # 1. Extract Thermal Features
        if have_th:
            t1, t2, t3, t4 = self.enc_th(thermal)
        
        # 2. Extract Event Features
        if have_ev:
            e1, e2, e3, e4 = self.enc_ev(events)

        # 3. Handle missing modalities (Zero padding)
        if have_th and not have_ev:
            e1, e2, e3, e4 = torch.zeros_like(t1), torch.zeros_like(t2), torch.zeros_like(t3), torch.zeros_like(t4)
        if have_ev and not have_th:
            t1, t2, t3, t4 = torch.zeros_like(e1), torch.zeros_like(e2), torch.zeros_like(e3), torch.zeros_like(e4)

        # 4. Dual Fusion 
        f1 = self.f1(t1, e1)
        f2 = self.f2(t2, e2)
        f3 = self.f3(t3, e3)
        f4 = self.f4(t4, e4)

        # 5. Decode 
        out = self.dec(f1, f2, f3, f4)
        return out


    @property
    def fusion_alpha_means(self):
        return (
            (self.f1.alpha_mean.item() if self.f1.alpha_mean is not None else float('nan')),
            (self.f2.alpha_mean.item() if self.f2.alpha_mean is not None else float('nan')),
            (self.f3.alpha_mean.item() if self.f3.alpha_mean is not None else float('nan')),
            (self.f4.alpha_mean.item() if self.f4.alpha_mean is not None else float('nan')),
        )


