
__all__ = [
    "Data_Pipeline",
    "Data_Pipeline_Trunc",
]

from . import Data_Pipeline_Functions as dpf
from .Data_Classes import datasets, _default_config_dir
import yaml
import os

def Data_Pipeline(dataset_name, config_dir=None):
    """Run the full analysis pipeline for *dataset_name*.

    Parameters
    ----------
    dataset_name : str
        Key of the dataset defined in ``datasets.yaml``.
    config_dir : str | None
        Directory containing ``datasets.yaml`` and
        ``analysis_parameters.yaml``.  Defaults to the package's built-in
        ``src/General_Data_Analysis/config/`` directory.  Pass a custom
        directory to use your own configuration files.
    """
    if config_dir is None:
        config_dir = _default_config_dir()

    # Validate dataset
    dataset = dpf.validate_dataset(dataset_name)

    # Load analysis parameters YAML
    with open(os.path.join(config_dir, 'analysis_parameters.yaml'), 'r') as file:
        analysis_params = yaml.safe_load(file)
    params = analysis_params[dataset_name]
    params = dpf.AnalysisParameters(params)

    
    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)
    dpf.Generic_Image_Processing(dataset,params.bound_list,idx=params.idx,thresh_1=params.thresh_1)

    if dataset_name == "January_2024_571":
        case_no = 1
    else:
        case_no = 0

    dpf.Background_Treatment(dataset,case_no=case_no)

    dpf.filter_beams(dataset,params.thresh,params.bg_thresh,params.proj_thresh)

    dpf.Generic_Moment_Calculation(dataset)

    dpf.Generic_VCC_Analysis(dataset,params.VCC_bound_list,params.VCC_idx)

    
def Data_Pipeline_Trunc(dataset_name):


    # Validate dataset
    dataset = dpf.validate_dataset(dataset_name)


    
    dpf.Generic_Preprocessing(dataset)
    dpf.Generic_DAQ_Preprocessing(dataset)
    dpf.Generic_Data_Processing(dataset)
    

    
