"""Teach compute_occlusion.load_lightning_model about the fusion architecture.

The existing implementation probes ``model.classifier.0.weight`` to decide which
classifier to build. SmilesAnchorResidualAuxClassifier checkpoints have no such
key (they carry base_proj / base_classifier / aux_proj / aux_classifier /
gate_mlp), so occlusion dies with a KeyError before doing any work.

This rewrites the function to dispatch on which keys are actually present.
Legacy checkpoints take exactly the path they took before.

Run once, from the repo root:

    python patch_loader.py

Writes compute_occlusion.py.bak first.
"""
from __future__ import annotations

from pathlib import Path
import shutil

TARGET = Path("src/downstream/toxicity/compute_occlusion.py")

OLD = '''    encoder = MolEncoderWrapper(encoder_config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    classifier_input_size = int(checkpoint["state_dict"]["model.classifier.0.weight"].shape[0])
    if classifier_input_size == encoder.hidden_size:
        classifier = LegacyClsToxicityClassifier(encoder=encoder, num_labels=2)
    else:
        classifier = ToxicityClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=len(config.token_feature_names),
            molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
        )
'''

NEW = '''    encoder = MolEncoderWrapper(encoder_config)
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]

    if "model.base_classifier.weight" in state:
        # SmilesAnchorResidualAuxClassifier: confidence-gated residual fusion.
        # Feature dimensions are read back from tensor shapes rather than taken
        # from the CLI, so a mismatch fails loudly instead of scoring wrongly.
        from downstream.toxicity.finetune import SmilesAnchorResidualAuxClassifier

        hidden = int(state["model.base_classifier.weight"].shape[1])
        ckpt_token_dim = 0
        if "model.token_proj.0.weight" in state:
            ckpt_token_dim = (int(state["model.token_proj.0.weight"].shape[0]) - hidden) // 4
        ckpt_molecule_dim = 0
        if "model.molecule_proj.1.weight" in state:
            ckpt_molecule_dim = int(state["model.molecule_proj.1.weight"].shape[1])

        want_token_dim = len(config.token_feature_names)
        want_molecule_dim = molecule_feature_dimension(config.molecule_feature_names)
        if ckpt_token_dim != want_token_dim:
            raise ValueError(
                f"{config.checkpoint_path}: checkpoint token_feature_dim="
                f"{ckpt_token_dim} but --token-features implies {want_token_dim}"
            )
        if ckpt_molecule_dim != want_molecule_dim:
            raise ValueError(
                f"{config.checkpoint_path}: checkpoint molecule_feature_dim="
                f"{ckpt_molecule_dim} but --molecule-features implies {want_molecule_dim}"
            )

        classifier = SmilesAnchorResidualAuxClassifier(
            encoder=encoder,
            num_labels=2,
            token_feature_dim=ckpt_token_dim,
            molecule_feature_dim=ckpt_molecule_dim,
            weight_mode=getattr(config, "token_weight_mode", "abs"),
        )
    else:
        classifier_input_size = int(state["model.classifier.0.weight"].shape[0])
        if classifier_input_size == encoder.hidden_size:
            classifier = LegacyClsToxicityClassifier(encoder=encoder, num_labels=2)
        else:
            classifier = ToxicityClassifier(
                encoder=encoder,
                num_labels=2,
                token_feature_dim=len(config.token_feature_names),
                molecule_feature_dim=molecule_feature_dimension(config.molecule_feature_names),
            )
'''


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} not found -- run this from the repo root")

    text = TARGET.read_text()

    if "model.base_classifier.weight" in text:
        print("already patched; nothing to do")
        return
    if OLD not in text:
        raise SystemExit(
            "could not find the block to replace -- compute_occlusion.py differs "
            "from what was inspected. Patch load_lightning_model by hand."
        )

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
    TARGET.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {TARGET}")
    print(f"backup at {TARGET.with_suffix('.py.bak')}")
    print("\nverify with:")
    print("  python -c \"from downstream.toxicity.compute_occlusion import load_lightning_model; print('ok')\"")


if __name__ == "__main__":
    main()
