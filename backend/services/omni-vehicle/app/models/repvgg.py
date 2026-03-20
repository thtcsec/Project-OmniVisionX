"""
RepVGG Backbone for LPR v3
Highly optimized backbone with structural re-parameterization
Perfect for TensorRT optimization
- Better accuracy than VGG/ResNet
- Can be converted to single 3x3 branch for inference
- 4th Gen Tensor Core friendly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict


class RepVGGBlock(nn.Module):
    """
    RepVGG building block with re-parameterization capability
    Train with multiple branches, infer with single 3x3 branch
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1, 
                 groups: int = 1, deploy: bool = False, use_se: bool = False):
        """
        Args:
            deploy: If True, use single 3x3 branch (inference mode)
            use_se: Squeeze-and-Excitation module
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy
        self.use_se = use_se
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            # 3x3 branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # 1x1 branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, padding=0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # Identity branch (only if in_channels == out_channels and stride == 1)
            if in_channels == out_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None
        
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            out = self.rbr_reparam(x)
            if self.se is not None:
                out = self.se(out)
            return F.relu(out)
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        if self.se is not None:
            out = self.se(out)
        return F.relu(out)
    
    def _fuse_bn_tensor(self, branch: nn.Sequential) -> torch.Tensor:
        """Fuse BN into Conv for re-parameterization"""
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                                       dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        """Convert trained model to deploy (inference) mode"""
        if self.deploy:
            return
        
        kernel, bias = self._fuse_bn_tensor(self.rbr_dense)
        
        if self.rbr_1x1 is not None:
            k1, b1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernel = kernel + F.pad(k1, [self.kernel_size // 2] * 4)
            bias = bias + b1
        
        if self.rbr_identity is not None:
            k_id, b_id = self._fuse_bn_tensor(self.rbr_identity)
            kernel = kernel + k_id
            bias = bias + b_id
        
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        
        for para in self.parameters():
            para.detach_()
        
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        
        self.deploy = True


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepVGG(nn.Module):
    """
    RepVGG backbone for LPR
    Lite version optimized for license plate recognition
    """
    
    def __init__(self, num_blocks: List[int], width_multiplier: List[float],
                 num_classes: int = 256, deploy: bool = False, use_se: bool = True):
        """
        Args:
            num_blocks: Number of blocks in each stage [2, 4, 14, 1]
            width_multiplier: Width multiplier for each stage [0.75, 0.75, 1.0, 2.5]
            num_classes: Output feature dimension
            deploy: Deploy mode (single 3x3 branch)
            use_se: Use SE blocks
        """
        super().__init__()
        self.deploy = deploy
        self.num_blocks = num_blocks
        self.width_multiplier = width_multiplier
        
        assert len(num_blocks) == 4
        assert len(width_multiplier) == 4
        
        self.in_planes = int(64 * width_multiplier[0])
        
        self.stage0 = self._make_stage(
            3, self.in_planes, num_blocks[0],
            stride=2, width_mult=width_multiplier[0],
            deploy=deploy, use_se=use_se
        )
        
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), int(128 * width_multiplier[1]),
            num_blocks[1], stride=2, width_mult=width_multiplier[1],
            deploy=deploy, use_se=use_se
        )
        
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), int(256 * width_multiplier[2]),
            num_blocks[2], stride=2, width_mult=width_multiplier[2],
            deploy=deploy, use_se=use_se
        )
        
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), int(512 * width_multiplier[3]),
            num_blocks[3], stride=2, width_mult=width_multiplier[3],
            deploy=deploy, use_se=use_se
        )
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        
        self._init_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int,
                   stride: int, width_mult: float, deploy: bool, use_se: bool) -> nn.Sequential:
        """Create stage with multiple RepVGG blocks"""
        blocks = []
        
        # First block with stride
        blocks.append(RepVGGBlock(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            deploy=deploy, use_se=use_se
        ))
        
        # Rest of blocks
        for _ in range(num_blocks - 1):
            blocks.append(RepVGGBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1,
                deploy=deploy, use_se=use_se
            ))
        
        return nn.Sequential(*blocks)
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage3 spatial feature map (B, C, H, W) without GAP/linear.

        Used by CTC heads which need 4D spatial features.
        """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input image (batch, 3, H, W)
            return_features: If True, return intermediate features
        
        Returns:
            output: Feature vector (batch, num_classes)
        """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        
        return x
    
    def switch_to_deploy(self):
        """Convert all blocks to deploy mode"""
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True


def repvgg_a0_lpr(deploy: bool = False, **kwargs) -> RepVGG:
    """RepVGG-A0 for LPR (Lite version)"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        deploy=deploy,
        **kwargs
    )


def repvgg_a1_lpr(deploy: bool = False, **kwargs) -> RepVGG:
    """RepVGG-A1 for LPR (Medium version)"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.0, 1.0, 1.0, 2.5],
        deploy=deploy,
        **kwargs
    )


def repvgg_a2_lpr(deploy: bool = False, **kwargs) -> RepVGG:
    """RepVGG-A2 for LPR (Standard version)"""
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        deploy=deploy,
        **kwargs
    )


# For testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = repvgg_a1_lpr(num_classes=256, deploy=False)
    model = model.to(device)
    
    # Test forward
    x = torch.randn(4, 3, 32, 128, device=device)
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Convert to deploy mode
    print("\nConverting to deploy mode...")
    model.switch_to_deploy()
    
    # Test inference
    output = model(x)
    print(f"Deploy output: {output.shape}")
