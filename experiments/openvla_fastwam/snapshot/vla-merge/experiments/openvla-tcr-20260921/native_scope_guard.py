"""Explicit native-path exception; never silently intersect a regression scope."""
PREFIX = 'backbone.vision_backbone.fused_featurizer.attn_pool.'
UNUSED_POOL_LINEARS = {PREFIX + suffix for suffix in ('q', 'kv', 'proj', 'mlp.fc1', 'mlp.fc2')}


def validate_native_scope(registered, observed):
    registered, observed = set(registered), set(observed)
    if observed - registered:
        raise ValueError('Trace contains unregistered modules')
    if registered - observed != UNUSED_POOL_LINEARS:
        raise ValueError('Unexplained registered versus observed Linear difference')
    return sorted(UNUSED_POOL_LINEARS)
