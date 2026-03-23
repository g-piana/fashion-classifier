# src/taxonomy.py

import yaml
from pathlib import Path


def load_normalizer(yaml_path: Path) -> dict[str, str]:
    """
    Returns a flat dict: raw_term_lowercase -> canonical_label
    Built by inverting the YAML structure.
    """
    with open(yaml_path) as f:
        taxonomy = yaml.safe_load(f)

    normalizer = {}
    for canonical, aliases in taxonomy.items():
        if canonical.startswith("_"):
            continue
        for alias in (aliases or []):
            normalizer[alias.lower()] = canonical
        # canonical maps to itself
        normalizer[canonical.lower()] = canonical

    return normalizer


def normalize_term(raw: str, normalizer: dict[str, str],
                   fallback: str = "_unknown") -> str:
    return normalizer.get(raw.strip().lower(), fallback)