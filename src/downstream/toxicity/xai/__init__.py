"""Attribution utilities for descriptor-augmented MolEncoder models.

This package is additive: nothing here modifies the training code. Every module
can be run stand-alone as ``python -m downstream.toxicity.xai.<module> --help``.

Modules
-------
common
    SMILES character <-> atom index alignment shared by all attribution code.
ckpt_peek
    Inspect a checkpoint: fusion scales, descriptor configuration, metrics.
compare_runs
    Side-by-side comparison of cross-validation results across runs.
selftest
    Self-tests for the character-walk and atom-index alignment logic.

The integrated-gradients pipeline used in the manuscript lives in ``scripts/xai/``.
"""

__all__ = [
    "common",
    "ckpt_peek",
    "compare_runs",
    "selftest",
]
