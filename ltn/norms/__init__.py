import ltn.norms.norms_config

# default configuration
norms_config.set_tnorm("luk")
norms_config.set_universal_aggregator('hmean')
norms_config.set_existential_aggregator('max')
