import os

# Paths to your YAML configuration files.
# Copy config/example_datasets.yaml and config/example_analysis_parameters.yaml
# as starting points.
DATASETS_YAML = "datasets.yaml"
ANALYSIS_PARAMETERS_YAML = "analysis_parameters.yaml"

Dataset_list = [
            # "March_2024_241",
            # "April_2024_241",
            # "April_2024_571",
            # "October_2024_571",
            "January_2024_241"
            ]

from General_Data_Analysis import Data_Pipeline

for dataset_name in Dataset_list:
    Data_Pipeline(dataset_name,
                  datasets_yaml=DATASETS_YAML,
                  analysis_parameters_yaml=ANALYSIS_PARAMETERS_YAML)
