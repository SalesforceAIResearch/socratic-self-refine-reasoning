from .module import (
    self_consistency,
    atom,
    mctsr,
    self_refine,
    debate,
)

from .module_ssr import (
    socratic_self_refine,
    socratic_self_refine_adaptive,
    socratic_self_refine_planning,
)

METHOD_HYPERPARAMS = {
    "cot": [],
    "sc": [
        ("num_samples", int, 5, "Number of samples for self-consistency"),
    ],
    "atom": [
        ("max_iters", int, 3, "Maximum number of iterations for the atom"),
    ],
    "mctsr": [
        ("max_iter", int, 4, "Maximum number of iterations for the MCTS-R"),
    ],
    "self-refine": [
        ("max_iter", int, 3, "Maximum number of iterations for the self-refine"),
    ],
    "debate": [
        ("max_iter", int, 3, "Maximum number of iterations for the debate"),
        ("num_agents", int, 2, "Number of agents for the debate"),
    ],
    "ssr": [
        ("max_iter", int, 3, "Maximum number of iterations for the socratic self-refine"),
    ],
    "ssr-adaptive": [
        ("max_iter", int, 3, "Maximum number of iterations for the socratic self-refine"),
    ],
    "ssr-planning": [
        ("max_iter", int, 3, "Maximum number of iterations for the socratic self-refine"),
    ],
}

METHOD_CONFIGS = {
    "cot": self_consistency,
    "sc": self_consistency,
    "atom": atom,
    "mctsr": mctsr,
    "self-refine": self_refine,
    "debate": debate,
    "ssr": socratic_self_refine,
    "ssr-adaptive": socratic_self_refine_adaptive,
    "ssr-planning": socratic_self_refine_planning,
    "ssr-planning-linear-scale": socratic_self_refine_planning,
}
