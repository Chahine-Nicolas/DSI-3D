from .dsi import DSI
from .dsi_void import DSI_VOID

__all__ = {
    'DSI_VOID': DSI_VOID,
    'DSI': DSI
}

def build_dsi(model_cfg, num_class, dataset, logger):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger
    )
    return model
