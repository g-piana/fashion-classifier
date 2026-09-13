# Adding a New Domain
## Complete cookbook for onboarding a new fashion category

This document is the single reference for adding a new domain
(e.g. dresses, bags, trousers) to the pipeline. Follow these steps
in order. No existing files need to be modified — everything is additive.

---

## What "domain" means

A domain is a top-level fashion category with its own:
- Taxonomy file (`taxonomies/<domain>.yaml`)
- One or more trained checkpoints (`weights/<model_name>/<run>/`)
- Optional brand fine-tunes (`weights/<model_name>/<run>_<brand>/`)

Examples: `shoes`, `jackets`, `dresses`, `bags`, `trousers`

---

## Step 1 — Define the taxonomy

Create `taxonomies/<domain>.yaml`. Use `taxonomies/shoes.yaml` as the
reference for a multi-stage domain and `taxonomies/jackets.yaml` for
a single-stage domain.

### Single-stage domain (category only, no subcategories yet)

```yaml
# taxonomies/dresses.yaml
domain: dresses
image_size: 224
backbone: resnet50

stage1:
  model_name: dresses_category
  run: "01"
  npy_run: "01"
  label_column: Class
  csv_file: labels_dresses_category.csv
  classes:
    - cocktail-dress
    - evening-gown
    - maxi-dress
    - midi-dress
    - mini-dress
    - shirt-dress
    - wrap-dress

stage2_map: {}   # add subcategory models here when training data is ready
```

### Multi-stage domain (category + subcategories)

```yaml
# taxonomies/bags.yaml
domain: bags
image_size: 224
backbone: resnet50

stage1:
  model_name: bags_category
  run: "01"
  npy_run: "01"
  label_column: Class
  csv_file: labels_bags_category.csv
  classes:
    - backpack
    - clutch
    - crossbody
    - shoulder-bag
    - tote

stage2_map:

  shoulder-bag:
    model_name: shoulder_bag_sub
    run: "01"
    npy_run: "01"
    label_column: Class
    csv_file: labels_shoulder_bag_sub.csv
    classes:
      - bucket-bag
      - flap-bag
      - hobo
      - saddle-bag
      - structured-bag

  # Add other categories as subcategory models are trained
```

**Rules for class lists:**
- Order must match the order in `conf/category/<model_name>.yaml` exactly
  (training config and taxonomy must agree — the manifest will freeze this)
- Use lowercase-with-hyphens consistently
- No trailing spaces or invisible characters

---

## Step 2 — Create Hydra category configs for training

For each model in the taxonomy, create `conf/category/<model_name>.yaml`.
The class list here must exactly match the taxonomy.

```yaml
# conf/category/dresses_category.yaml
name: "dresses_category"
label_column: "Class"
label_type: "single"
csv_file: "labels_dresses_category.csv"
classes:
  - "cocktail-dress"
  - "evening-gown"
  - "maxi-dress"
  - "midi-dress"
  - "mini-dress"
  - "shirt-dress"
  - "wrap-dress"
```

Also create `conf/data/<model_name>.yaml` for each model:

```yaml
# conf/data/dresses_category.yaml
defaults:
  - base

run: "01"
raw_subdir: "dresses/photo"
image_size: 224
max_frames: 40000
```

---

## Step 3 — Prepare training data

Collect images and produce a labels CSV at:
`E:/fashion-data/csv/labels_<model_name>.csv`

Required columns: `name` (image stem, no extension), `Class` (or
whatever label_column is set to).

Place raw images at:
`E:/fashion-data/01-RAW/<raw_subdir>/`

---

## Step 4 — Preprocess and train

Run once per model in the taxonomy.

```powershell
# Preprocess (only needed once per image set)
python src/preprocess.py `
    category=dresses_category `
    data=dresses_category `
    filesystem=local

# Train
python src/train.py `
    category=dresses_category `
    model=resnet50 `
    training=default `
    data=dresses_category `
    filesystem=local
# → writes weights/dresses_category/01/best.ckpt
# → writes weights/dresses_category/01/normalization.npy
# → writes weights/dresses_category/01/manifest.json  ← automatic
```

For subsequent runs with different hyperparameters (same images):

```powershell
python src/train.py `
    category=dresses_category `
    model=resnet50 `
    training=default `
    data=dresses_category `
    data.run=02 `
    data.npy_run=01 `    # reuse existing npy
    filesystem=local
```

---

## Step 5 — Validate taxonomy resolution

```powershell
python src/registry.py dresses --weights-root "E:/fashion-data/weights"
```

Expected output:
```
Resolving checkpoints for domain 'dresses'
  stage1  : dresses_category/01  classes=[...]
✓  All checkpoints resolved for domain 'dresses'
```

If any manifest is missing, either retrain (generates it automatically)
or run the backfill script:

```powershell
python scripts/backfill_manifests.py
```

---

## Step 6 — Run the pipeline

```powershell
# Full run
python src/pipeline.py `
    --domain    dresses `
    --image-dir "E:/fashion-data/01-RAW/new_batch/dresses" `
    --out-csv   "E:/fashion-data/csv/pipeline_dresses.csv"

# Admin upload path — category known upfront
python src/pipeline.py `
    --domain           dresses `
    --image-dir        "E:/fashion-data/01-RAW/new_batch/dresses" `
    --out-csv          "E:/fashion-data/csv/pipeline_dresses.csv" `
    --known-category   "maxi-dress"
```

---

## Adding subcategories to an existing domain

If the domain is already in production and you want to add subcategory
models later:

1. Train the subcategory model (Steps 2-4 for the new model only)
2. Add the `stage2_map` entry to `taxonomies/<domain>.yaml`
3. Update `run` in the taxonomy to point to the new checkpoint
4. Re-validate with `registry.py`

No changes to `pipeline.py`, `registry.py`, or any other code.

---

## Adding a brand fine-tune

Train from the base checkpoint with a brand-scoped output run:

```powershell
python src/train.py `
    category=dresses_category `
    model=resnet50 `
    training=finetune `
    data=dresses_category `
    data.run=01_gucci `      # convention: {base_run}_{brand}
    data.npy_run=01 `
    training.finetune_from=01 `
    filesystem=local
```

The registry resolves this automatically when `--brand gucci` is passed:

```powershell
python src/pipeline.py `
    --domain    dresses `
    --image-dir "E:/fashion-data/01-RAW/gucci_fw26/dresses" `
    --out-csv   "E:/fashion-data/csv/pipeline_dresses_gucci.csv" `
    --brand     gucci
```

---

## Checklist for a new domain

- [ ] `taxonomies/<domain>.yaml` created
- [ ] `conf/category/<model_name>.yaml` created for each model
- [ ] `conf/data/<model_name>.yaml` created for each model
- [ ] Labels CSV prepared at `csv/labels_<model_name>.csv`
- [ ] Raw images placed at `01-RAW/<raw_subdir>/`
- [ ] `preprocess.py` run for each model
- [ ] `train.py` run for each model
- [ ] `registry.py <domain>` validation passes
- [ ] `pipeline.py --domain <domain> --dry-run` passes
- [ ] `pipeline.py --domain <domain>` produces correct output CSV
