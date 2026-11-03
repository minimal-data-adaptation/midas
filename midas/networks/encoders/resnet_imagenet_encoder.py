"""Pretrained ImageNet ResNet-50 encoder for pixel-based RL.

Wraps HuggingFace's FlaxResNetModule so ResNet-50 parameters live in the
learner's TrainState and receive gradients. Pretrained ImageNet-1k weights
are loaded separately via `load_resnet_imagenet_pretrained_params()` and
injected into the TrainState after initialization.

BatchNorm is forced into eval mode (`use_running_average=True`) so no
mutable `batch_stats` collection is produced at apply time. This keeps
the existing residual-learner JIT update graphs unchanged. Gradients
still flow through conv kernels and BN affine scale/bias.

Preprocessing replicates HuggingFace's AutoImageProcessor pipeline for
microsoft/resnet-50 using JAX ops (JIT-compatible):
  1. Rescale uint8 [0,255] → float [0,1]
  2. Resize shortest edge to 256
  3. Center crop to 224×224
  4. Normalize with ImageNet mean/std
  5. Transpose to channels-first (B, C, H, W) — FlaxResNetModule convention
"""

from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoImageProcessor, ResNetConfig
from transformers.models.resnet.modeling_flax_resnet import FlaxResNetModule


class ResNetImageNetEncoder(nn.Module):
    """ImageNet-pretrained ResNet encoder matching the repo's encoder interface.

    Input:  uint8 observations with shape (B, H, W, C, 1)
    Output: (B, hidden_size) — pooled backbone feature (2048 for ResNet-50)

    Uses the internal FlaxResNetModule so parameters are part of the enclosing
    Flax param tree. Pretrained weights must be loaded after TrainState init
    via load_resnet_imagenet_pretrained_params().
    """
    model_name: str = "microsoft/resnet-50"

    def setup(self):
        config = ResNetConfig.from_pretrained(self.model_name)
        self.resnet_imagenet = FlaxResNetModule(config)
        self.hidden_size = config.hidden_sizes[-1]

        proc = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        size = proc.size
        if "shortest_edge" in size:
            self.resize_shortest_edge = size["shortest_edge"]
        else:
            self.resize_shortest_edge = size.get("height", 224)

        crop = getattr(proc, "crop_size", None) or size
        self.crop_h = crop.get("height", 224)
        self.crop_w = crop.get("width", 224)
        self.image_mean = jnp.array(proc.image_mean)
        self.image_std = jnp.array(proc.image_std)

    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1]))  # (B, H, W, C)
        batch_size, h, w, c = x.shape

        scale = self.resize_shortest_edge / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        x = jax.image.resize(x, shape=(batch_size, new_h, new_w, c), method="bilinear")

        top = (new_h - self.crop_h) // 2
        left = (new_w - self.crop_w) // 2
        x = jax.lax.dynamic_slice(
            x, (0, top, left, 0), (batch_size, self.crop_h, self.crop_w, c)
        )

        x = (x - self.image_mean) / self.image_std

        # HF FlaxResNet expects channels-last (B, H, W, C). BN always runs in
        # eval mode via deterministic=True, so no mutable batch_stats are
        # produced regardless of the outer `train` flag.
        outputs = self.resnet_imagenet(x, deterministic=True)
        feat = outputs.pooler_output
        # Pooler output may be (B, C, 1, 1) or (B, 1, 1, C); flatten all trailing dims
        feat = jnp.reshape(feat, (feat.shape[0], int(np.prod(feat.shape[1:]))))
        return feat


def load_resnet_imagenet_pretrained_params(
    model_name: str = "microsoft/resnet-50",
) -> Dict[str, Any]:
    """Load pretrained ResNet-50 ImageNet-1k weights from the HuggingFace hub.

    Returns a dict with keys:
      - "params": the trainable param subtree, ready to replace the
        'resnet_imagenet' key in the encoder's param tree.
      - "batch_stats": the BN running statistics subtree, or an empty dict
        if the installed transformers version folds them into `params`.
    """
    from transformers import FlaxResNetModel

    model = FlaxResNetModel.from_pretrained(model_name, from_pt=True)
    # HF's FlaxResNetModel stores variables as a single dict with both
    # 'params' and 'batch_stats' nested inside model.params.
    vars_dict = dict(model.params)
    params_subtree = vars_dict.get("params", vars_dict)
    batch_stats_subtree = vars_dict.get("batch_stats", {}) or {}
    return {
        "params": dict(params_subtree),
        "batch_stats": dict(batch_stats_subtree),
    }
