import t_norms as tnorms

TRIANGULAR_NORMS = {
    'min': {
        'AND': tnorms.min_and,
        'OR': tnorms.min_or,
        'NOT': tnorms.min_not,
        'IMPLIES': tnorms.min_implies,
        'EQUIVALENT': tnorms.min_equivalent
    },

    'luk': {
        'AND': tnorms.luk_and,
        'OR': tnorms.luk_or,
        'NOT': tnorms.luk_not,
        'IMPLIES': tnorms.luk_implies,
        'EQUIVALENT': tnorms.luk_equivalent
    },

    'prod': {
        'AND': tnorms.luk_and,
        'OR': tnorms.luk_or,
        'NOT': tnorms.luk_not,
        'IMPLIES': tnorms.luk_implies,
        'EQUIVALENT': tnorms.luk_equivalent,

    },

    'mean': {
        'AND': tnorms.mean_and,
        'OR': tnorms.mean_or,
        'NOT': tnorms.mean_not,
        'IMPLIES': tnorms.mean_implies,
        'EQUIVALENT': tnorms.mean_equivalent,
    },

    # #####  FOR FOL #####

    'universal': {
        'hmean': tnorms.hmean_universal_aggregation,
        'min': tnorms.min_universal_aggregation,
        'mean': tnorms.mean_universal_aggregation,
    },

    'existence': {
        'max': tnorms.max_existence_aggregation,
    }
}