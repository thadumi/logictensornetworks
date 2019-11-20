from .t_norms import TRIANGULAR_NORMS

F_And = None
F_Or = None
F_Not = None

F_Implies = None
F_Equiv = None

F_ForAll = None
F_Exists = None


def set_tnorm(tnorm_kind: str):
    global F_And, F_Or, F_Implies, F_Not, F_Equiv
    assert tnorm_kind in TRIANGULAR_NORMS.keys()

    F_Or = TRIANGULAR_NORMS[tnorm_kind]['OR']
    F_And = TRIANGULAR_NORMS[tnorm_kind]['AND']
    F_Not = TRIANGULAR_NORMS[tnorm_kind]['NOT']
    F_Equiv = TRIANGULAR_NORMS[tnorm_kind]['EQUIVALENT']
    F_Implies = TRIANGULAR_NORMS[tnorm_kind]['IMPLIES']


def set_universal_aggregator(aggregator_kind: str):
    global F_ForAll
    assert aggregator_kind in TRIANGULAR_NORMS['universal'].keys()

    F_ForAll = TRIANGULAR_NORMS['universal'][aggregator_kind]


def set_existential_aggregator(aggregator_kind: str):
    global F_Exists
    assert aggregator_kind in TRIANGULAR_NORMS['existence'].keys()

    F_Exists = TRIANGULAR_NORMS['existence'][aggregator_kind]
