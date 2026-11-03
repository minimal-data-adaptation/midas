import jax

from flax.core.frozen_dict import FrozenDict

from midas.types import Params

from functools import partial

def soft_target_update(critic_params: Params, target_critic_params: Params, tau: float) -> Params:
    new_target_params = jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params)
    return new_target_params


def soft_target_update_skip_encoder(critic_params: Params, target_critic_params: Params, tau: float) -> Params:
    """Soft target EMA that leaves the top-level ``encoder`` subtree untouched.

    Used when the vision encoder is frozen: live and target critics carry
    identical encoder params, so the EMA over those leaves is a no-op that
    still walks the full subtree (~86M DINOv2 leaves) every critic update
    and allocates a fresh copy. Routing the encoder subtree around the EMA
    avoids both costs while keeping head/Dense/LN params on the standard
    EMA path.
    """
    def _ema(p, tp):
        return p * tau + tp * (1 - tau)

    head_new = {k: v for k, v in critic_params.items() if k != 'encoder'}
    head_target = {k: v for k, v in target_critic_params.items() if k != 'encoder'}
    head_out = jax.tree_util.tree_map(_ema, head_new, head_target)
    out = dict(head_out)
    if 'encoder' in target_critic_params:
        out['encoder'] = target_critic_params['encoder']
    if isinstance(target_critic_params, FrozenDict):
        return FrozenDict(out)
    return out

@partial(jax.pmap, axis_name='pmap', static_broadcasted_argnums=(2))
def soft_target_update_parallel(critic_params: Params, target_critic_params: Params, tau: float) -> Params:
    new_target_params = jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params)
    return new_target_params
