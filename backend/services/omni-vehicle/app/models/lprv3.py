"""
LPRv3 End-to-End Model
Fortress Edition with TPS-STN + RepVGG + 2-Line CTC
Tối ưu hóa cho RTX 5090 với Mixed Precision + TensorRT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import logging

from .tps_stn import TPSSTN
from .repvgg import repvgg_a1_lpr
from .ctc_decoder import TwoLineCTCDecoder

logger = logging.getLogger(__name__)


class LPRv3(nn.Module):
    """
    Complete LPR v3 system
    1. TPS-STN: Normalize plate geometry
    2. RepVGG Backbone: Extract features
    3. CTC Heads: Decode top and bottom lines separately
    """

    def __init__(self,
                 num_classes: int = 256,
                 tps_output_size: Tuple[int, int] = (32, 100),
                 vocab_top: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-",
                 vocab_bottom: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                 use_mixed_precision: bool = True):
        super().__init__()

        self.tps_output_size = tps_output_size
        self.vocab_top = vocab_top
        self.vocab_bottom = vocab_bottom
        self.use_mixed_precision = use_mixed_precision

        # Stage 1: Spatial normalization
        self.stn = TPSSTN(
            in_channels=3,
            num_fiducial=16,
            tps_grid_size=4,
            tps_output_size=tps_output_size
        )

        # Stage 2: Feature extraction
        self.backbone = repvgg_a1_lpr(num_classes=num_classes, deploy=False, use_se=True)

        # Determine backbone output channels dynamically
        # repvgg_a1_lpr: width_multiplier=[1,1,1,2.5] → stage3 = int(512*2.5) = 1280
        backbone_out_ch = int(512 * 2.5)  # 1280 for repvgg_a1_lpr

        # Stage 3: CTC Decoders (separate heads for top and bottom lines)
        self.ctc_head_top = nn.Sequential(
            nn.Conv2d(backbone_out_ch, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(vocab_top), kernel_size=1)
        )

        self.ctc_head_bottom = nn.Sequential(
            nn.Conv2d(backbone_out_ch, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, len(vocab_bottom), kernel_size=1)
        )

        # CTC decoder
        self.decoder = TwoLineCTCDecoder(vocab_top=vocab_top, vocab_bottom=vocab_bottom)

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize CTC heads"""
        for m in [self.ctc_head_top, self.ctc_head_bottom]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        batch_size = x.shape[0]

        # Stage 1: TPS-STN normalization
        warped, grid = self.stn(x)

        # Stage 2: Extract spatial feature maps (single pass)
        # forward_features() returns the stage3 conv output (B,C,H,W)
        # without GAP/linear — needed by CTC heads.
        x_feat = self.backbone.forward_features(warped)

        # Also compute the pooled feature vector for the output dict
        features = self.backbone.linear(
            self.backbone.gap(x_feat).view(batch_size, -1)
        )

        # CTC heads
        ctc_top = self.ctc_head_top(x_feat)
        ctc_bottom = self.ctc_head_bottom(x_feat)

        # Reshape for CTC: (T, batch, vocab_size)
        ctc_top = ctc_top.permute(3, 0, 2, 1).reshape(-1, batch_size, ctc_top.shape[1])
        ctc_bottom = ctc_bottom.permute(3, 0, 2, 1).reshape(-1, batch_size, ctc_bottom.shape[1])

        # Logging softmax for CTC loss
        ctc_top = F.log_softmax(ctc_top, dim=-1)
        ctc_bottom = F.log_softmax(ctc_bottom, dim=-1)

        output = {
            'output_top': ctc_top,
            'output_bottom': ctc_bottom,
            'features': features,
            'warped_image': warped,
        }

        return output

    def infer(self, x: torch.Tensor, beam_width: int = 5,
             use_fp16: bool = False) -> List[Dict]:
        self.eval()

        with torch.no_grad():
            if use_fp16 and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    output = self.forward(x)
            else:
                output = self.forward(x)

        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            ctc_top = output['output_top'][:, b, :]
            ctc_bottom = output['output_bottom'][:, b, :]

            # Apply exp to convert from log_softmax back to probabilities
            ctc_top = torch.exp(ctc_top)
            ctc_bottom = torch.exp(ctc_bottom)

            plate_result = self.decoder.decode(
                ctc_top,
                ctc_bottom,
                beam_width=beam_width,
                use_beam_search=True
            )

            results.append(plate_result)

        return results

    def compute_loss(self, output: Dict, target_top: torch.Tensor,
                    target_bottom: torch.Tensor,
                    input_lengths: torch.Tensor,
                    target_lengths_top: torch.Tensor,
                    target_lengths_bottom: torch.Tensor) -> torch.Tensor:
        ctc_top = output['output_top']
        ctc_bottom = output['output_bottom']

        loss_top = self.ctc_loss(ctc_top, target_top, input_lengths, target_lengths_top)
        loss_bottom = self.ctc_loss(ctc_bottom, target_bottom, input_lengths, target_lengths_bottom)

        # Weighted sum
        total_loss = 0.5 * loss_top + 0.5 * loss_bottom

        return total_loss

    def switch_to_deploy(self):
        """Convert to inference-optimized mode"""
        logger.info("Converting model to deploy mode...")

        # Convert RepVGG to single 3x3 branch
        self.backbone.switch_to_deploy()

        self.eval()
        logger.info("✅ Model ready for TensorRT export")

    def export_onnx(self, output_path: str = "lprv3_fortress.onnx",
                   example_input_size: Tuple[int, int, int] = (1, 3, 64, 128)):
        self.switch_to_deploy()
        self.eval()

        device = next(self.parameters()).device
        example_input = torch.randn(example_input_size, device=device)

        logger.info("Exporting to ONNX: %s", output_path)

        torch.onnx.export(
            self,
            example_input,
            output_path,
            input_names=['input'],
            output_names=['output_top', 'output_bottom', 'features', 'warped_image'],
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )
