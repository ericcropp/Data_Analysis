import numpy as np
import pytest
import yaml

from General_Data_Analysis.Data_Pipeline import Data_Pipeline


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
def smoke_env(tmp_path):
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
    config_dir = tmp_path / "config"
    config_dir.mkdir()

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

    # -- analysis_parameters.yaml ----------------------------------------
    fake_params = {
        'test_571_smoke': {
            'bound_list': [{'xstart': 100, 'xend': 600, 'ystart': 100, 'yend': 600}],
            'idx': 0,
            'thresh': 0,
            'bg_thresh': 6,
            'proj_thresh': 4e5,
            'VCC_bound_list': [{'xstart': 250, 'xend': 1050, 'ystart': 100, 'yend': 900}],
            'VCC_idx': 0,
            'thresh_1': 1e6,
        }
    }
    analysis_parameters_yaml = config_dir / 'analysis_parameters.yaml'
    with open(str(analysis_parameters_yaml), 'w') as fh:
        yaml.dump(fake_params, fh)

    # -- datasets.yaml ----------------------------------------------------
    empty_keys = [
        'CAMR:LT10:900:Image:ArrayData',
        'CAMR:LT10:900:XRMS',
        'CAMR:LT10:900:YRMS',
        'CAMR:LT10:900:Image:ArraySize1_RBV',
        'CAMR:LT10:900:Image:ArraySize0_RBV',
        'CAMR:LT10:900:X',
        'CAMR:LT10:900:Y',
        'CAMR:LT10:900:RESOLUTION',
        'PROF:IN10:241:Image:ArrayData',
        'PROF:IN10:241:XRMS',
        'PROF:IN10:241:YRMS',
        'PROF:IN10:241:Image:ArraySize1_RBV',
        'PROF:IN10:241:Image:ArraySize0_RBV',
        'PROF:IN10:241:X',
        'PROF:IN10:241:Y',
        'PROF:IN10:241:RESOLUTION',
        'PROF:IN10:571:Image:ArrayData',
        'PROF:IN10:571:XRMS',
        'PROF:IN10:571:YRMS',
        'PROF:IN10:571:Image:ArraySize1_RBV',
        'PROF:IN10:571:Image:ArraySize0_RBV',
        'PROF:IN10:571:X',
        'PROF:IN10:571:Y',
        'PROF:IN10:571:RESOLUTION',
        'PROF:IN10:711:Image:ArrayData',
        'PROF:IN10:711:XRMS',
        'PROF:IN10:711:YRMS',
        'PROF:IN10:711:Image:ArraySize1_RBV',
        'PROF:IN10:711:Image:ArraySize0_RBV',
        'PROF:IN10:711:X',
        'PROF:IN10:711:Y',
        'PROF:IN10:711:RESOLUTION',
    ]
    datasets_cfg = {
        'paths': {'s3df': '', 'NERSC': ''},
        'empty_keys': empty_keys,
        'datasets': {
            'test_571_smoke': {
                'pathlist': [str(data_dir) + '/'],
                'screen': screen,
                'save_loc': str(save_dir) + '/',
                'bg_file': str(bg_dir) + '/*background*',
                'raw_vcc': str(vcc_dir) + '/vcc*.npy',
            }
        },
    }
    datasets_yaml = config_dir / 'datasets.yaml'
    with open(str(datasets_yaml), 'w') as fh:
        yaml.dump(datasets_cfg, fh)

    return str(datasets_yaml), str(analysis_parameters_yaml)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_Data_Pipeline(smoke_env):
    datasets_yaml, analysis_parameters_yaml = smoke_env
    Data_Pipeline("test_571_smoke",
                  datasets_yaml=datasets_yaml,
                  analysis_parameters_yaml=analysis_parameters_yaml)
