import numpy as np
import pytest
import os

from General_Data_Analysis.Data_Pipeline import Data_Pipeline
from General_Data_Analysis.Data_Classes import Data_Set, datasets


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_gaussian_image(nrows, ncols, cx, cy, sigma_x, sigma_y,
                         amplitude=1000, baseline=5):
    """Return a 2-D Gaussian as a **flat** numpy array."""
    y, x = np.mgrid[0:nrows, 0:ncols]
    img = amplitude * np.exp(
        -((x - cx) ** 2 / (2 * sigma_x ** 2)
          + (y - cy) ** 2 / (2 * sigma_y ** 2))
    )
    img += baseline + np.abs(np.random.normal(0, 0.5, img.shape))
    return img.flatten()


# ---------------------------------------------------------------------------
# fixture – builds the whole synthetic environment in a tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def smoke_env(tmp_path, monkeypatch):
    screen = 'PROF:IN10:571:Image:ArrayData'
    nrows, ncols = 700, 700
    cx, cy = 350, 350
    sigma_x, sigma_y = 30, 25
    n_shots = 5

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    vcc_dir = tmp_path / "vcc"
    vcc_dir.mkdir()

    # -- synthetic value / image .npy files (per shot) --------------------
    for i in range(n_shots):
        ts = f"2025-01-27T12-00-0{i}-08-00"      # exactly 25 chars
        assert len(ts) == 25

        # scalar PVs
        values_dict = {
            'PROF:IN10:571:Image:ArraySize1_RBV': nrows,
            'PROF:IN10:571:Image:ArraySize0_RBV': ncols,
            'PROF:IN10:571:XRMS': 30.0,
            'PROF:IN10:571:YRMS': 25.0,
            'PROF:IN10:571:X': 350.0,
            'PROF:IN10:571:Y': 350.0,
            'PROF:IN10:571:RESOLUTION': 10.0,
        }

        # image PVs
        flat_img = _make_gaussian_image(nrows, ncols, cx, cy,
                                        sigma_x, sigma_y)
        imgs_dict = {
            screen: flat_img,
            'PROF:IN10:571:Image:ArraySize1_RBV': nrows,
            'PROF:IN10:571:Image:ArraySize0_RBV': ncols,
            'PROF:IN10:571:XRMS': 30.0,
            'PROF:IN10:571:YRMS': 25.0,
            'PROF:IN10:571:X': 350.0,
            'PROF:IN10:571:Y': 350.0,
            'PROF:IN10:571:RESOLUTION': 10.0,
        }

        np.save(str(data_dir / f"data_values_{ts}.npy"), values_dict)
        np.save(str(data_dir / f"data_imgs_{ts}.npy"), imgs_dict)

    # -- background .npy file ---------------------------------------------
    bg_dict = {
        screen: _make_gaussian_image(nrows, ncols, cx, cy,
                                     sigma_x, sigma_y,
                                     amplitude=5, baseline=2),
        'PROF:IN10:571:Image:ArraySize1_RBV': nrows,
        'PROF:IN10:571:Image:ArraySize0_RBV': ncols,
    }
    np.save(str(bg_dir / "background_001.npy"), bg_dict)

    # -- VCC .npy file ----------------------------------------------------
    vcc_n = 1100
    vcc_dict = {
        'CAMR:LT10:900:Image:ArrayData':
            _make_gaussian_image(vcc_n, vcc_n, 550, 550, 100, 80),
        'CAMR:LT10:900:Image:ArraySize1_RBV': vcc_n,
        'CAMR:LT10:900:Image:ArraySize0_RBV': vcc_n,
    }
    np.save(str(vcc_dir / "vcc_001.npy"), vcc_dict)

    # -- empty dict (keys that may be missing from imgs) ------------------
    empty = {
        'CAMR:LT10:900:Image:ArrayData': '',
        'CAMR:LT10:900:XRMS': '',
        'CAMR:LT10:900:YRMS': '',
        'CAMR:LT10:900:Image:ArraySize1_RBV': '',
        'CAMR:LT10:900:Image:ArraySize0_RBV': '',
        'CAMR:LT10:900:X': '',
        'CAMR:LT10:900:Y': '',
        'CAMR:LT10:900:RESOLUTION': '',
        'PROF:IN10:241:Image:ArrayData': '',
        'PROF:IN10:241:XRMS': '',
        'PROF:IN10:241:YRMS': '',
        'PROF:IN10:241:Image:ArraySize1_RBV': '',
        'PROF:IN10:241:Image:ArraySize0_RBV': '',
        'PROF:IN10:241:X': '',
        'PROF:IN10:241:Y': '',
        'PROF:IN10:241:RESOLUTION': '',
        'PROF:IN10:571:Image:ArrayData': '',
        'PROF:IN10:571:XRMS': '',
        'PROF:IN10:571:YRMS': '',
        'PROF:IN10:571:Image:ArraySize1_RBV': '',
        'PROF:IN10:571:Image:ArraySize0_RBV': '',
        'PROF:IN10:571:X': '',
        'PROF:IN10:571:Y': '',
        'PROF:IN10:571:RESOLUTION': '',
        'PROF:IN10:711:Image:ArrayData': '',
        'PROF:IN10:711:XRMS': '',
        'PROF:IN10:711:YRMS': '',
        'PROF:IN10:711:Image:ArraySize1_RBV': '',
        'PROF:IN10:711:Image:ArraySize0_RBV': '',
        'PROF:IN10:711:X': '',
        'PROF:IN10:711:Y': '',
        'PROF:IN10:711:RESOLUTION': '',
    }

    # -- build the Data_Set -----------------------------------------------
    test_paths = {'s3df': '', 'NERSC': ''}
    dataset = Data_Set(
        pathlist=[str(data_dir) + '/'],
        screen=screen,
        save_loc=str(save_dir) + '/',
        paths=test_paths,
        computer='s3df',
        empty=empty,
        prefixes=None,
        DAQ_Matching=None,
        bg_file=str(bg_dir) + '/*background*',
        raw_vcc=str(vcc_dir) + '/vcc*.npy',
    )

    # patch the module-level dict so Data_Pipeline finds our dataset
    monkeypatch.setitem(datasets, 'test_571_smoke', dataset)
    return dataset


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_Data_Pipeline(smoke_env):
    Data_Pipeline("test_571_smoke")
