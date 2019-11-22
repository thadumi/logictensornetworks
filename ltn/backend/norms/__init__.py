import norms_config

# default configuration
norms_config.set_tnorm("luk")
norms_config.set_universal_aggregator('hmean')
norms_config.set_existential_aggregator('max')

from norms_config import F_Exists, F_Equiv, F_Not, F_Implies, F_Or, F_And, F_ForAll