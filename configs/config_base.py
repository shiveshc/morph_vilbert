from ml_collections import ConfigDict


def transformer_cfg() -> ConfigDict:
    '''
    Base Config for morph_mm.models.transformer.Transformer
    '''
    cfg = ConfigDict()
    cfg.model_dim = 32
    cfg.num_heads = 4
    cfg.mlp_inner_dim = 64
    cfg.num_layers = 3
    return cfg


def mlp_decoder_cfg() -> ConfigDict:
    '''
    Base Config for morph_mm.models.decoder.MLPDecoder
    '''
    cfg = ConfigDict()
    cfg.mlp_inner_dim = 64
    cfg.mlp_num_inner_layers = 1
    return cfg

