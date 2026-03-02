import numpy as np
import pytest
import pandas as pd
import types

from General_Data_Analysis.DAQ_Extract import DAQ_1D_Extraction_v2

def make_fake_mat_struct():
    # Simulate the structure returned by loadmat
    class Dummy:
        pass

    # Simulate scan steps and values
    scan_vals = np.array([1, 2, 3, 4, 5])
    scan_steps = np.array([1, 2, 3, 4, 5])
    scan_PV = 'PV1'
    cam_names = ['CAM1', 'CAM2']
    num_CAM = 2

    # Simulate pulseID and scalars
    common_index = np.array([1, 2, 3, 4, 5])
    common_index_data = np.array([1, 2, 3, 4, 5])

    # Simulate images
    files = ['scan_fake_CAM1_1.tif', 'scan_fake_CAM1_2.tif', 'scan_fake_CAM1_3.tif', 'scan_fake_CAM1_4.tif', 'scan_fake_CAM1_5.tif']
    images = {
        'CAM1': {'loc': files},
        'CAM2': {'loc': files}
    }

    # Simulate BSA lists
    BSA_List_S11 = np.random.rand(5, 2)
    BSA_List_S10 = np.random.rand(5, 2)
    BSA_List_S10RF = np.random.rand(5, 2)

    # Simulate pulseID for cameras
    pulseID = {
        'common_scalar_index': common_index,
        'CAM1common_index': common_index,
        'CAM2common_index': common_index
    }

    # Build the fake structure
    data_struct = {
        'params': {
            'scanPVs': scan_PV,
            'scanVals': scan_vals,
            'camNames': cam_names,
            'num_CAM': num_CAM
        },
        'scalars': {
            'steps': scan_steps,
            'common_index': common_index_data,
            'BSA_List_S11': BSA_List_S11,
            'BSA_List_S10': BSA_List_S10,
            'BSA_List_S10RF': BSA_List_S10RF
        },
        'pulseID': pulseID,
        'images': images
    }

    return {'data_struct': data_struct}


def fake_loadmat(filepath):
    return make_fake_mat_struct()

def fake_imageio_imread(path):
    # Return a 10x10 array for any image
    return np.ones((10, 10))

@pytest.mark.unit
def test_DAQ_1D_Extraction_v2(monkeypatch):
    # Patch loadmat and imageio.imread
    monkeypatch.setattr("General_Data_Analysis.DAQ_Extract.loadmat", fake_loadmat)
    monkeypatch.setattr("imageio.imread", fake_imageio_imread)

    # Run extraction
    data = DAQ_1D_Extraction_v2('/fake/path', 'scan_fake')

    # Check output DataFrame
    assert isinstance(data, pd.DataFrame)
    assert 'CAM1_files' in data.columns
    assert 'CAM2_files' in data.columns
    assert 'CAM1_images' in data.columns
    assert 'CAM2_images' in data.columns
    assert len(data['CAM1_images'].iloc[0]) == 10 * 10 or isinstance(data['CAM1_images'].iloc[0], np.ndarray)
