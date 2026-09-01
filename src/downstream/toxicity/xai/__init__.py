"""XAI diagnostics and improved attribution for descriptor-augmented MolEncoder models.

This package is additive: nothing here modifies the original training or
occlusion code. Every module can be run stand-alone as
``python -m downstream.toxicity.xai.<module> --help``.

Modules
-------
common
    SMILES character <-> atom index alignment shared by all attribution code.
baselines
    Empirical reference values for descriptors (median / mean / marginal),
    used instead of the out-of-distribution zero baseline.
diagnostics
    Does the trained model actually use the descriptors? Reads the checkpoint
    and reports fusion scales and per-block classifier weight mass.
coverage
    How many molecules really got semiempirical values instead of silently
    falling back to Gasteiger / Crippen approximations.
ablation
    Random / shuffled / noisy / scaled descriptor ablations on a split.
counterfactual_occlusion
    Token occlusion that recomputes molecule-level descriptors after the
    perturbation, plus baseline-aware global descriptor occlusion.
faithfulness
    Deletion / insertion AUC, comprehensiveness, sufficiency, and stability
    of attributions under SMILES re-randomisation.
wins
    Which molecules the descriptor-augmented model fixes relative to the
    SMILES-only baseline, and what they have in common.
"""

__all__ = [
    "common",
    "baselines",
    "diagnostics",
    "coverage",
    "ablation",
    "counterfactual_occlusion",
    "faithfulness",
    "wins",
]
