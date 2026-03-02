Dataset_list = [
            # "March_2024_241",
            # "April_2024_241",
            # "April_2024_571",
            # "October_2024_571",
            "January_2025_241",
            "January_2025_571",
            ]
from General_Data_Analysis import Data_Pipeline
for dataset_name in Dataset_list:
    Data_Pipeline(dataset_name)
