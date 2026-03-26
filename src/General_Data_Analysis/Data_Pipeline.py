
__all__ = [
    "Data_Pipeline",
    "Data_Pipeline_Trunc",
]

from . import Data_Pipeline_Functions as dpf
from .Data_Classes import load_datasets
import yaml
import os

def Data_Pipeline(dataset_name, datasets_yaml, analysis_parameters_yaml):
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
    datasets = load_datasets(datasets_yaml)

    # Validate dataset
    dataset = dpf.validate_dataset(dataset_name, datasets)

    # Load analysis parameters YAML
    with open(analysis_parameters_yaml, 'r') as file:
        analysis_params = yaml.safe_load(file)
    params = analysis_params[dataset_name]
    params = dpf.AnalysisParameters(params)

    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)
    dpf.Generic_Image_Processing(dataset, params.bound_list, idx=params.idx, thresh_1=params.thresh_1)

    if dataset_name == "January_2024_571":
        case_no = 1
    else:
        case_no = 0

    dpf.Background_Treatment(dataset, case_no=case_no)

    dpf.filter_beams(dataset, params.thresh, params.bg_thresh, params.proj_thresh)

    dpf.Generic_Moment_Calculation(dataset)

    dpf.Generic_VCC_Analysis(dataset, params.VCC_bound_list, params.VCC_idx)


def Data_Pipeline_Trunc(dataset_name, datasets_yaml):
    """Run the truncated pipeline (preprocessing only) for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Key of the dataset defined in ``datasets_yaml``.
    datasets_yaml : str
        Path to the datasets YAML file (copy
        ``config/example_datasets.yaml`` as a starting point).
    """
    datasets = load_datasets(datasets_yaml)

    # Validate dataset
    dataset = dpf.validate_dataset(dataset_name, datasets)

    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)
