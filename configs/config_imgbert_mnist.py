from ml_collections import ConfigDict

from configs.config_base import transformer_cfg, mlp_decoder_cfg


def get_train_cfg() -> ConfigDict:
    
    data_config = ConfigDict()
    data_config.in_channels = 1
    data_config.img_size = 28
    data_config.patch_size = 4

    model_config = ConfigDict()
    model_config.encoder = transformer_cfg()
    model_config.decoder = mlp_decoder_cfg()
    model_config.mask_ratio = 0.25

    cfg = ConfigDict()
    cfg.data = data_config
    cfg.model = model_config
    cfg.loss = 'l2'

    return cfg


