
__all__ = [
    "Data_Pipeline",
    "Data_Pipeline_Trunc",
]

from . import Data_Pipeline_Functions as dpf
from .Data_Classes import load_datasets, _detect_computer
import yaml
import os


# ── config archive helpers ──────────────────────────────────────────────────

def _check_archive_exists(base_output, dataset_name, check_params=True):
    """Return True if *dataset_name* is already present in the archived config.

    Parameters
    ----------
    base_output : str
        Base output directory to inspect.
    dataset_name : str
        Dataset key to look for.
    check_params : bool
        Also check ``analysis_parameters.yaml`` (default ``True``).
    """
    datasets_path = os.path.join(base_output, 'datasets.yaml')
    if os.path.exists(datasets_path):
        with open(datasets_path, 'r') as fh:
            existing = yaml.safe_load(fh) or {}
        if dataset_name in existing.get('datasets', {}):
            return True

    if check_params:
        params_path = os.path.join(base_output, 'analysis_parameters.yaml')
        if os.path.exists(params_path):
            with open(params_path, 'r') as fh:
                existing = yaml.safe_load(fh) or {}
            if dataset_name in existing:
                return True

    return False


def _archive_config(raw_datasets_cfg, analysis_params_raw, dataset_name):
    """Merge *dataset_name*'s config into the archived YAML files in base_output.

    Parameters
    ----------
    raw_datasets_cfg : dict
        Parsed contents of the input datasets YAML.
    analysis_params_raw : dict | None
        Parsed contents of the input analysis_parameters YAML, or ``None``
        when archiving for ``Data_Pipeline_Trunc`` (no params needed).
    dataset_name : str
        Dataset key whose config should be archived.
    """
    base_output = raw_datasets_cfg['paths'][_detect_computer()]
    os.makedirs(base_output, exist_ok=True)

    # ── Archive datasets.yaml ───────────────────────────────────────────────
    datasets_path = os.path.join(base_output, 'datasets.yaml')
    if os.path.exists(datasets_path):
        with open(datasets_path, 'r') as fh:
            archived_datasets = yaml.safe_load(fh) or {}
    else:
        archived_datasets = {
            'paths': raw_datasets_cfg['paths'],
            'empty_keys': raw_datasets_cfg['empty_keys'],
            'datasets': {},
            'aliases': {},
        }

    # Merge (or overwrite) this dataset's entry
    archived_datasets.setdefault('datasets', {})[dataset_name] = \
        raw_datasets_cfg['datasets'][dataset_name]

    # Merge any aliases that point to this dataset
    archived_datasets.setdefault('aliases', {})
    for alias, target in raw_datasets_cfg.get('aliases', {}).items():
        if target == dataset_name:
            archived_datasets['aliases'][alias] = target

    with open(datasets_path, 'w') as fh:
        yaml.dump(archived_datasets, fh, default_flow_style=False)

    # ── Archive analysis_parameters.yaml ────────────────────────────────────
    if analysis_params_raw is not None:
        params_path = os.path.join(base_output, 'analysis_parameters.yaml')
        if os.path.exists(params_path):
            with open(params_path, 'r') as fh:
                archived_params = yaml.safe_load(fh) or {}
        else:
            archived_params = {}

        archived_params[dataset_name] = analysis_params_raw[dataset_name]

        with open(params_path, 'w') as fh:
            yaml.dump(archived_params, fh, default_flow_style=False)


# ── pipeline entry points ───────────────────────────────────────────────────

def Data_Pipeline(dataset_name, datasets_yaml, analysis_parameters_yaml,
                  _confirm=None):
    """Run the full analysis pipeline for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Key of the dataset defined in ``datasets_yaml``.
    datasets_yaml : str
        Path to the datasets YAML file (copy
        ``config/example_datasets.yaml`` as a starting point).
    analysis_parameters_yaml : str
        Path to the analysis parameters YAML file (copy
        ``config/example_analysis_parameters.yaml`` as a starting point).
    """
    # ── Read raw configs ────────────────────────────────────────────────────
    with open(datasets_yaml, 'r') as fh:
        raw_datasets_cfg = yaml.safe_load(fh) or {}
    with open(analysis_parameters_yaml, 'r') as fh:
        analysis_params_raw = yaml.safe_load(fh) or {}

    # ── Pre-run overwrite check ─────────────────────────────────────────────
    base_output = raw_datasets_cfg.get('paths', {}).get(_detect_computer(), '')
    if base_output and _check_archive_exists(base_output, dataset_name,
                                             check_params=True):
        msg = (f"Dataset '{dataset_name}' already has archived config in "
               f"{base_output!r}. Overwrite? [y/N]: ")
        approved = _confirm(msg) if _confirm is not None else \
            input(msg).strip().lower() == 'y'
        if not approved:
            print(f"Aborted: will not overwrite existing config for "
                  f"'{dataset_name}'.")
            return

    # ── Build in-memory objects ─────────────────────────────────────────────
    datasets = load_datasets(datasets_yaml)
    dataset = dpf.validate_dataset(dataset_name, datasets)
    params = dpf.AnalysisParameters(analysis_params_raw[dataset_name])

    # ── Pipeline steps ──────────────────────────────────────────────────────
    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)
    dpf.Generic_Image_Processing(dataset, params.bound_list, idx=params.idx,
                                 thresh_1=params.thresh_1)

    if dataset_name == "January_2024_571":
        case_no = 1
    else:
        case_no = 0

    dpf.Background_Treatment(dataset, case_no=case_no)
    dpf.filter_beams(dataset, params.thresh, params.bg_thresh, params.proj_thresh)
    dpf.Generic_Moment_Calculation(dataset)
    dpf.Generic_VCC_Analysis(dataset, params.VCC_bound_list, params.VCC_idx)

    # ── Post-run: archive config ────────────────────────────────────────────
    if base_output:
        _archive_config(raw_datasets_cfg, analysis_params_raw, dataset_name)


def Data_Pipeline_Trunc(dataset_name, datasets_yaml, _confirm=None):
    """Run the truncated pipeline (preprocessing only) for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Key of the dataset defined in ``datasets_yaml``.
    datasets_yaml : str
        Path to the datasets YAML file (copy
        ``config/example_datasets.yaml`` as a starting point).
    """
    # ── Read raw config ─────────────────────────────────────────────────────
    with open(datasets_yaml, 'r') as fh:
        raw_datasets_cfg = yaml.safe_load(fh) or {}

    # ── Pre-run overwrite check (datasets only) ─────────────────────────────
    base_output = raw_datasets_cfg.get('paths', {}).get(_detect_computer(), '')
    if base_output and _check_archive_exists(base_output, dataset_name,
                                             check_params=False):
        msg = (f"Dataset '{dataset_name}' already has archived config in "
               f"{base_output!r}. Overwrite? [y/N]: ")
        approved = _confirm(msg) if _confirm is not None else \
            input(msg).strip().lower() == 'y'
        if not approved:
            print(f"Aborted: will not overwrite existing config for "
                  f"'{dataset_name}'.")
            return

    # ── Build in-memory objects ─────────────────────────────────────────────
    datasets = load_datasets(datasets_yaml)
    dataset = dpf.validate_dataset(dataset_name, datasets)

    # ── Pipeline steps ──────────────────────────────────────────────────────
    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)

    # ── Post-run: archive datasets config only ──────────────────────────────
    if base_output:
        _archive_config(raw_datasets_cfg, None, dataset_name)
