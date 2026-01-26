

import ml_collections
import torch

def get_config():
  config = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()


  data.image_size = 320
  data.num_channels = 2

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000 
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'positional'
  model.name = 'ncsnpp'
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config

