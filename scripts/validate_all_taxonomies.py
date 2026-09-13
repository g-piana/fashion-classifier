"""
scripts/validate_all_taxonomies.py
===================================
Validates that every taxonomy file in taxonomies/ resolves all its
checkpoints correctly. Run after training new models or adding domains.

Usage:
    python scripts/validate_all_taxonomies.py
    python scripts/validate_all_taxonomies.py --domain shoes
    python scripts/validate_all_taxonomies.py --weights-root "E:/fashion-data/weights"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from registry import load_taxonomy, resolve_taxonomy_checkpoints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain",       default=None,
                   help="Validate a single domain only (default: all)")
    p.add_argument("--taxonomies",   type=Path, default=Path("taxonomies"),
                   help="Path to taxonomies/ directory")
    p.add_argument("--weights-root", type=Path,
                   default=Path("E:/fashion-data/weights"))
    args = p.parse_args()

    taxonomy_files = (
        [args.taxonomies / f"{args.domain}.yaml"]
        if args.domain
        else sorted(args.taxonomies.glob("*.yaml"))
    )

    if not taxonomy_files:
        print(f"No taxonomy files found in {args.taxonomies}")
        sys.exit(1)

    print(f"Validating {len(taxonomy_files)} taxonomy file(s)\n")

    passed = []
    failed = []

    for tf in taxonomy_files:
        domain = tf.stem
        try:
            taxonomy = load_taxonomy(domain, args.taxonomies)
            resolved = resolve_taxonomy_checkpoints(taxonomy, args.weights_root)
            print(f"  ✓  {domain}")
            print(f"       stage1  : {resolved.stage1.model_name}/{resolved.stage1.run}"
                  f"  classes={resolved.stage1.classes}")
            for cat, sub in resolved.stage2_map.items():
                print(f"       stage2  : {sub.model_name}/{sub.run}"
                      f"  [{cat}]")
            passed.append(domain)
        except Exception as e:
            print(f"  ✗  {domain}")
            print(f"       {e}")
            failed.append(domain)
        print()

    print(f"{'='*50}")
    print(f"Passed : {len(passed)}  {passed}")
    print(f"Failed : {len(failed)}  {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
