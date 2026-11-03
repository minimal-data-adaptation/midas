"""DINOv2 Vision Transformer encoder for pixel-based RL.

Wraps HuggingFace's FlaxDinov2Module (the internal Flax nn.Module) so that
DINOv2 parameters live in the learner's TrainState and receive gradients.
Pretrained weights are loaded separately via `load_dinov2_pretrained_params()`
and injected into the TrainState after initialization.

Preprocessing replicates HuggingFace's AutoImageProcessor pipeline for
facebook/dinov2-base using JAX ops (JIT-compatible):
  1. Rescale uint8 [0,255] → float [0,1]
  2. Resize shortest edge to 256
  3. Center crop to 224×224
  4. Normalize with ImageNet mean/std

Default: ViT-B/14, 768-dim [CLS] output, crop_size=224.
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import AutoImageProcessor, Dinov2Config
from transformers.models.dinov2.modeling_flax_dinov2 import FlaxDinov2Module


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder matching the interface of other encoders in this repo.

    Input:  uint8 observations with shape (B, H, W, C, 1)
    Output: (B, hidden_size) from the [CLS] pooler output

    Uses the internal FlaxDinov2Module so parameters are part of the enclosing
    Flax param tree. Pretrained weights must be loaded after TrainState init
    via load_dinov2_pretrained_params().

    Preprocessing parameters (resize target, crop size, mean, std) are read
    from the model's AutoImageProcessor at setup time, then applied with JAX
    ops inside __call__ so the full forward pass is JIT-compatible.
    """
    model_name: str = "facebook/dinov2-base"

    def setup(self):
        config = Dinov2Config.from_pretrained(self.model_name)
        self.dinov2 = FlaxDinov2Module(config)
        self.hidden_size = config.hidden_size

        # Read preprocessing config from HF image processor
        proc = AutoImageProcessor.from_pretrained(self.model_name)
        self.resize_shortest_edge = proc.size["shortest_edge"]  # 256
        self.crop_h = proc.crop_size["height"]  # 224
        self.crop_w = proc.crop_size["width"]   # 224
        self.image_mean = jnp.array(proc.image_mean)  # [0.485, 0.456, 0.406]
        self.image_std = jnp.array(proc.image_std)     # [0.229, 0.224, 0.225]

    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))  # (B, H, W, C)
        batch_size, h, w, c = x.shape

        # Resize shortest edge to self.resize_shortest_edge (default 256).
        # Use Python-level arithmetic so new_h/new_w are concrete integers —
        # jax.image.resize requires a concrete shape tuple inside JIT.
        scale = self.resize_shortest_edge / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        x = jax.image.resize(x, shape=(batch_size, new_h, new_w, c), method="bilinear")

        # Center crop to (crop_h, crop_w)
        top = (new_h - self.crop_h) // 2
        left = (new_w - self.crop_w) // 2
        x = jax.lax.dynamic_slice(
            x, (0, top, left, 0), (batch_size, self.crop_h, self.crop_w, c)
        )

        # ImageNet normalization
        x = (x - self.image_mean) / self.image_std

        # FlaxDinov2Module expects channels-last (B, H, W, C)
        outputs = self.dinov2(x, deterministic=not train)
        return outputs.pooler_output  # (B, hidden_size)


def load_dinov2_pretrained_params(
    model_name: str = "facebook/dinov2-base",
) -> dict:
    """Load pretrained DINOv2 weights from the HuggingFace hub.

    Returns the params dict from the pretrained FlaxDinov2Model, which can be
    used to replace the 'dinov2' subtree in the encoder's param tree.

    Usage in learner __init__ after TrainState creation:
        pretrained = load_dinov2_pretrained_params("facebook/dinov2-base")
        # Replace encoder params in actor/critic states
    """
    from transformers import FlaxDinov2Model

    model = FlaxDinov2Model.from_pretrained(model_name, from_pt=True)
    return model.params
