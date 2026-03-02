# General Data Analysis

A Python package for FACET-II injector beam-physics data analysis. The pipeline
ingests raw `.npy` image/scalar files and `.mat` DAQ files, processes them
through a series of stages, and produces:

1. A **DataFrame** of all scalar measurements (consistent ordering)
2. An **image stack** aligned to that ordering
3. A **background image** for subtraction
4. **Second-order moments** (beam sizes) via configurable analysis methods
5. A **VCC image** suitable for distgen

## Project layout

```
General_Data_Analysis/
├── pyproject.toml                          # Build, deps & pytest config
├── README.md
├── environment.yml                         # Minimal conda environment
├── Examples/
│   └── Runner.py                           # Batch-run the pipeline
├── Manual_Analysis_Tools/                  # Step-by-step Jupyter notebooks
│   ├── 1_Generic_Preprocessing.ipynb
│   ├── 2_Generic_DAQ_Preprocessing.ipynb
│   ├── 3_Generic_Data_Processing.ipynb
│   ├── 4_Generic_Image_Processing.ipynb
│   ├── 5_Jan_2024_Background-Copy1.ipynb
│   ├── 5.1_Apr_2024_241_Background.ipynb
│   ├── 6_Filter_Beams.ipynb
│   ├── 7_Generic_Moment_Calculation.ipynb
│   └── 8_Generic_VCC_Analysis.ipynb
├── config/                                 # Example configs (copy & customize)
│   ├── example_datasets.yaml               # Sanitized dataset template
│   └── example_analysis_parameters.yaml    # Sanitized parameters template
├── src/General_Data_Analysis/              # Installable package
│   ├── __init__.py
│   ├── Data_Classes.py                     # Data_Set class + YAML loader
│   ├── Data_Pipeline.py                    # Top-level pipeline entry points
│   ├── Data_Pipeline_Functions.py          # Pipeline step implementations
│   ├── DAQ_Extract.py                      # MATLAB DAQ file extraction
│   ├── Image_Analysis.py                   # Gaussian fitting, RMS, cropping
│   └── config/                             # Default built-in configuration
│       ├── datasets.yaml                   # Dataset definitions (see above example --> copy here)
│       └── analysis_parameters.yaml        # Per-dataset analysis parameters (see above example --> copy here)
└── tests/
    ├── smoke/
    │   └── test_smoke.py                   # End-to-end with synthetic data
    └── unit/
        ├── test_DAQ.py
        ├── test_Data_Pipeline.py
        └── test_Image_Analysis.py
```

## Installation

### Recommended: conda environment

A minimal, fully-pinned environment is provided. From the repo root:

```bash
conda env create -f environment.yml
conda activate general_data_analysis
```

This installs all runtime and test dependencies and then installs the package
itself in editable mode (`pip install -e .`).

To update an existing environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Alternative: pip (editable install)

```bash
pip install -e .            # core dependencies
pip install -e ".[dev]"     # + pytest
```

### Dependencies

numpy · pandas · scipy · matplotlib · h5py · opencv-python · Pillow · imageio · PyYAML

## Quick start

### Automated pipeline

```python
from General_Data_Analysis import Data_Pipeline

Data_Pipeline("January_2025_571")
```

Or batch-run several datasets:

```python
from General_Data_Analysis import Data_Pipeline

for name in ["January_2025_241", "January_2025_571"]:
    Data_Pipeline(name)
```

See `Examples/Runner.py` for a ready-made script.

### Manual (notebook) workflow

The notebooks in `Manual_Analysis_Tools/` walk through each stage interactively.
Their primary purpose is **determining analysis parameters** (crop regions,
thresholds, background selection, etc.) that are then recorded in
`analysis_parameters.yaml` and consumed by the automated pipeline.

> **Source of truth:** If any behaviour in a notebook appears to differ from the
> automated pipeline, the implementations in `Data_Pipeline_Functions.py` are
> authoritative.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_Generic_Preprocessing` | Raw `.npy` → DataFrame |
| 2 | `2_Generic_DAQ_Preprocessing` | Raw `.mat` DAQ → DataFrame |
| 3 | `3_Generic_Data_Processing` | Merge the two DataFrames |
| 4 | `4_Generic_Image_Processing` | Build image stack, align, crop (**user input**) |
| 5 | `5_*_Background` | Assemble background when it must be pieced together |
| 6 | `6_Filter_Beams` | Remove bad beams by threshold filters |
| 7 | `7_Generic_Moment_Calculation` | Beam-size analysis (multiple methods) |
| 8 | `8_Generic_VCC_Analysis` | Produce a VCC image for distgen (**user input**) |

## Configuration

All configuration lives in two YAML files in `src/General_Data_Analysis/config/`
(the package default).  No Python edits are needed to add a new dataset or tune
analysis parameters.

To use **your own config files**, create a directory containing
`datasets.yaml` and `analysis_parameters.yaml` (copy from `config/example_*.yaml`
as a starting point), then pass it to the pipeline:

```python
Data_Pipeline("My_Dataset_571", config_dir="/path/to/my/config/")
```

### `datasets.yaml` — dataset definitions

Each entry under `datasets:` defines a measurement set. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `pathlist` | list of str | Directories (or YAML files) containing raw `.npy` data |
| `screen` | str | PV name of the screen image (e.g. `PROF:IN10:571:Image:ArrayData`) |
| `save_loc` | str | Sub-path appended to the site base path for saving outputs |

Optional fields (default to `null`):

| Field | Type | Description |
|-------|------|-------------|
| `prefixes` | list of str | DAQ scan prefixes to filter by |
| `DAQ_Matching` | str | Path to a YAML file mapping `.npy` files to DAQ scans |
| `bg_file` | str | Glob pattern for background image files |
| `raw_vcc` | str | Glob for VCC images, or `"included"` if embedded in data |

The file also contains:

- **`paths`** — base save directories for each compute site (`NERSC`, `s3df`).
  The active site is detected automatically from the hostname.
- **`empty_keys`** — the 32 PV keys tracked across all datasets (defined once).
- **`aliases`** — shorthand names that point to an existing dataset entry.

#### Adding a new dataset

Append a block to `datasets.yaml`:

```yaml
  My_New_Dataset:
    pathlist:
      - "/path/to/raw/data/"
    screen: "PROF:IN10:571:Image:ArrayData"
    save_loc: "2026-02-28/"
    bg_file: "/path/to/backgrounds/*background*"
    raw_vcc: "included"
```

Then add matching analysis parameters (see below).

If using a custom config directory, pass it when running the pipeline:

```python
Data_Pipeline("My_New_Dataset", config_dir="/path/to/my/config/")
```

### `analysis_parameters.yaml` — per-dataset analysis tuning

The `Manual_Analysis_Tools` notebooks are the intended way to determine these
parameters interactively. Each entry is keyed by the same dataset name used in
`datasets.yaml`.  See `config/example_analysis_parameters.yaml` for a
fully-commented template:

| Parameter | Description |
|-----------|-------------|
| `bound_list` | List of crop-region dicts (`xstart`, `xend`, `ystart`, `yend`) |
| `idx` | Which background image to use (`0`, `-1`, etc.) |
| `thresh` | Intensity threshold for beam filtering (removes clipped beams) |
| `bg_thresh` | Border-mean threshold for image filtering |
| `proj_thresh` | Projection-mean threshold for filtering |
| `thresh_1` | Secondary intensity threshold (BG selection; image processing stage) |
| `VCC_bound_list` | Crop regions for VCC analysis |
| `VCC_idx` | Which VCC image to use (irrelevant if `raw_vcc == "included"`, as shot-by-shot VCC images will be included) |

## Package API

Everything is importable from the top-level namespace:

```python
import General_Data_Analysis as gda

# Pipeline with custom config directory
gda.Data_Pipeline("My_Dataset_571", config_dir="/path/to/my/config/")

# Access a dataset object
ds = gda.datasets["January_2025_571"]
pathlist, screen, save_loc, empty, prefixes, DAQ_Matching, bg_file, raw_vcc = ds.return_params()

```

### Key modules

| Module | Exports |
|--------|---------|
| `Data_Pipeline` | `Data_Pipeline`, `Data_Pipeline_Trunc` |
| `Data_Classes` | `Data_Set`, `datasets` |
| `Data_Pipeline_Functions` | `AnalysisParameters`, `validate_dataset`, `Generic_Preprocessing`, `Generic_DAQ_Preprocessing`, `Generic_Data_Processing`, `Generic_Image_Processing`, `Background_Treatment`, `filter_beams`, `Generic_Moment_Calculation`, `Generic_VCC_Analysis`, plus utilities |
| `DAQ_Extract` | `DAQ_1D_Extraction_v2`, `loadmat` |
| `Image_Analysis` | `GaussianParams`, `Gaussian_Fit_4_Dim`, `RMS_Calc`, `RMS_Image_Analysis`, `image_cropp_center`, `ellipse_crop_v3`, `bg_thresh`, `imrotate45`, and more |

## Testing

```bash
pytest                  # run all tests
pytest -m unit          # unit tests only
pytest -m smoke         # end-to-end smoke test (synthetic data)
pytest -v               # verbose output
```

The smoke test creates synthetic Gaussian beam data in a temporary directory and
runs the full `Data_Pipeline` end-to-end, verifying that all stages complete
without error.

## Notes

- These scripts are intended for use on an **s3df or NERSC compute node**.
  Large datasets may exceed memory on login nodes.
- Notebook 4 requires **user input** at two steps: selecting crop regions and
  choosing a background image when one must be picked from the data.
- Notebook 8 requires **user input** to select the VCC image and crop rectangle.
