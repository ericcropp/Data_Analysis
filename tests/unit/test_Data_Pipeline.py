import yaml
import numpy as np
import pytest
import os
import tempfile
import h5py
import pandas as pd
import types

from General_Data_Analysis.Data_Pipeline_Functions import (
    screen_nickname_finder,
    second_parent,
    fix_path,
    read_hdf,
    Merge,
    im_stack_from_df,
    get_border_mean,
    get_proj_mean,
    get_x_y_proj_mean,
    safe_extract,
    rms_extractor,
    zscore,
    outlier_detect,
)

@pytest.mark.unit
def test_screen_nickname_finder(tmp_path):
    screen = 'CAMR:LT10:900:Image:ArrayData'
    save_loc = tmp_path / "testdir"
    nickname = screen_nickname_finder(screen, str(save_loc))
    assert nickname == '900'
    assert os.path.exists(save_loc)

@pytest.mark.unit
def test_second_parent():
    path = '/a/b/c/file.txt'
    assert second_parent(path) == '/a/b'

@pytest.mark.unit
def test_fix_path():
    new_path = '/new/parent/file.txt'
    old_path = '/old/parent/sub/file.txt'
    # Should replace old parent with new parent
    fixed = fix_path(new_path, old_path)
    assert '/new/parent/' in fixed

@pytest.mark.unit
def test_read_hdf():
    # Create a temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        with h5py.File(tmp.name, 'w') as f:
            dset = f.create_dataset('group1', data=np.arange(5))
        arr = read_hdf(tmp.name)
        assert np.all(arr == np.arange(5))

@pytest.mark.unit
def test_Merge():
    d1 = {'a': 1}
    d2 = {'b': 2}
    Merge(d1, d2)
    assert d1['b'] == 2

@pytest.mark.unit
def test_im_stack_from_df():
    df = pd.DataFrame({
        'img': [np.arange(6).reshape(2,3), np.arange(6,12).reshape(2,3)],
        'ncol': [2,2],
        'nrow': [3,3]
    })
    arr = im_stack_from_df(df, 'img', 'ncol', 'nrow')
    assert arr.shape == (2,2,3)
    assert np.all(arr[0] == np.arange(6).reshape(2,3))
    assert np.all(arr[1] == np.arange(6,12).reshape(2,3))


class DummyFit:
    def __init__(self, sigma):
        self.Sigma = sigma

def test_get_border_mean():
    img = np.ones((20, 20))
    img[0:10, :] = 2
    img[-10:, :] = 3
    img[10:-10, 0:10] = 4
    img[10:-10, -10:] = 5
    mean = get_border_mean(img, border_width=10)
    # Should be the mean of all border values
    assert np.isclose(mean, np.mean(np.concatenate([
        img[:10, :].ravel(),
        img[-10:, :].ravel(),
        img[10:-10, :10].ravel(),
        img[10:-10, -10:].ravel()
    ])))

def test_get_proj_mean():
    proj = np.arange(20)
    mean = get_proj_mean(proj, border_width=5)
    expected = np.mean(np.concatenate([proj[:5], proj[-5:]]))
    assert np.isclose(mean, expected)

def test_get_x_y_proj_mean():
    img = np.ones((10, 10))
    xmean, ymean = get_x_y_proj_mean(img)
    assert np.isclose(xmean, 10)
    assert np.isclose(ymean, 10)

def test_safe_extract():
    fit = DummyFit(5.5)
    assert safe_extract(fit) == 5.5
    class NoSigma: pass
    assert np.isnan(safe_extract(NoSigma()))

def test_rms_extractor():
    fits = [DummyFit(1), DummyFit(2), DummyFit(3)]
    arr = rms_extractor(fits)
    assert np.all(arr == np.array([1,2,3]))
    with pytest.raises(ValueError):
        rms_extractor(123)  # Not iterable

def test_zscore_and_outlier_detect():
    series = pd.Series([1,2,3,100])
    zs = zscore(series)
    assert len(zs) == 4
    print(zs)
    outliers = outlier_detect(zs, thresh=1.5)


# ---------------------------------------------------------------------------
# Tests for load_datasets
# ---------------------------------------------------------------------------

import yaml as _yaml
from General_Data_Analysis.Data_Classes import load_datasets, Data_Set

@pytest.mark.unit
def test_load_datasets_valid(tmp_path):
    """load_datasets returns a populated dict for a well-formed YAML."""
    datasets_yaml = tmp_path / "datasets.yaml"
    cfg = {
        'paths': {'NERSC': '/nersc/', 's3df': '/s3df/'},
        'empty_keys': ['KEY_A', 'KEY_B'],
        'datasets': {
            'My_Dataset': {
                'pathlist': ['/data/'],
                'screen': 'PROF:IN10:571:Image:ArrayData',
                'save_loc': 'output/',
            }
        },
    }
    with open(str(datasets_yaml), 'w') as fh:
        _yaml.dump(cfg, fh)

    result = load_datasets(str(datasets_yaml))
    assert 'My_Dataset' in result
    assert isinstance(result['My_Dataset'], Data_Set)

@pytest.mark.unit
def test_load_datasets_empty_yaml(tmp_path):
    """load_datasets returns an empty dict for an empty YAML file."""
    datasets_yaml = tmp_path / "datasets.yaml"
    datasets_yaml.write_text("")
    result = load_datasets(str(datasets_yaml))
    assert result == {}

@pytest.mark.unit
def test_load_datasets_aliases(tmp_path):
    """load_datasets resolves aliases to the same Data_Set object."""
    datasets_yaml = tmp_path / "datasets.yaml"
    cfg = {
        'paths': {'NERSC': '/nersc/', 's3df': '/s3df/'},
        'empty_keys': ['KEY_A'],
        'datasets': {
            'My_Dataset': {
                'pathlist': ['/data/'],
                'screen': 'PROF:IN10:571:Image:ArrayData',
                'save_loc': 'output/',
            }
        },
        'aliases': {'my_alias': 'My_Dataset'},
    }
    with open(str(datasets_yaml), 'w') as fh:
        _yaml.dump(cfg, fh)

    result = load_datasets(str(datasets_yaml))
    assert result['my_alias'] is result['My_Dataset']

@pytest.mark.unit
def test_load_datasets_missing_file():
    """load_datasets raises FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_datasets("/nonexistent/path/datasets.yaml")

@pytest.mark.unit
def test_Data_Pipeline_missing_datasets_yaml(tmp_path):
    """Data_Pipeline raises FileNotFoundError for a nonexistent datasets YAML."""
    from General_Data_Analysis import Data_Pipeline
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("")
    with pytest.raises(FileNotFoundError):
        Data_Pipeline("Any_Dataset",
                      datasets_yaml="/nonexistent/datasets.yaml",
                      analysis_parameters_yaml=str(params_yaml))

@pytest.mark.unit
def test_Data_Pipeline_missing_analysis_yaml(tmp_path):
    """Data_Pipeline raises FileNotFoundError for a nonexistent analysis YAML."""
    import yaml as _yaml
    from General_Data_Analysis import Data_Pipeline
    # Provide a valid (minimal) datasets.yaml so load_datasets succeeds
    datasets_yaml = tmp_path / "datasets.yaml"
    cfg = {
        'paths': {'NERSC': '/nersc/', 's3df': '/s3df/'},
        'empty_keys': ['KEY_A'],
        'datasets': {
            'My_Dataset': {
                'pathlist': ['/data/'],
                'screen': 'PROF:IN10:571:Image:ArrayData',
                'save_loc': 'output/',
            }
        },
    }
    with open(str(datasets_yaml), 'w') as fh:
        _yaml.dump(cfg, fh)
    with pytest.raises((FileNotFoundError, ValueError)):
        Data_Pipeline("My_Dataset",
                      datasets_yaml=str(datasets_yaml),
                      analysis_parameters_yaml="/nonexistent/params.yaml")


# ---------------------------------------------------------------------------
# Helpers shared by archive tests
# ---------------------------------------------------------------------------

def _minimal_datasets_cfg(base_output, dataset_name='DS', save_loc='run/'):
    return {
        'paths': {
            'NERSC': str(base_output) + '/',
            's3df': str(base_output) + '/',
        },
        'empty_keys': ['KEY_A'],
        'datasets': {
            dataset_name: {
                'pathlist': ['/data/'],
                'screen': 'PROF:IN10:571:Image:ArrayData',
                'save_loc': save_loc,
            }
        },
        'aliases': {},
    }


def _minimal_params_cfg(dataset_name='DS'):
    return {
        dataset_name: {
            'bound_list': [], 'idx': 0, 'thresh': 1,
            'bg_thresh': 1, 'proj_thresh': 1,
            'VCC_bound_list': [], 'VCC_idx': 0, 'thresh_1': 1.0,
        }
    }


# ---------------------------------------------------------------------------
# _check_archive_exists
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_check_archive_exists_false_when_no_files(tmp_path):
    """Returns False when no archived files exist."""
    from General_Data_Analysis.Data_Pipeline import _check_archive_exists
    assert not _check_archive_exists(str(tmp_path), 'DS')


@pytest.mark.unit
def test_check_archive_exists_true_after_datasets_archive(tmp_path):
    """Returns True after the datasets entry is archived."""
    from General_Data_Analysis.Data_Pipeline import _check_archive_exists, _archive_config
    cfg = _minimal_datasets_cfg(tmp_path)
    _archive_config(cfg, _minimal_params_cfg(), 'DS')
    assert _check_archive_exists(str(tmp_path) + '/', 'DS', check_params=False)


@pytest.mark.unit
def test_check_archive_exists_true_after_params_archive(tmp_path):
    """Returns True (via params check) after analysis_parameters is archived."""
    from General_Data_Analysis.Data_Pipeline import _check_archive_exists, _archive_config
    cfg = _minimal_datasets_cfg(tmp_path)
    _archive_config(cfg, _minimal_params_cfg(), 'DS')
    assert _check_archive_exists(str(tmp_path) + '/', 'DS', check_params=True)


@pytest.mark.unit
def test_check_archive_exists_false_for_different_name(tmp_path):
    """Returns False for a dataset not yet archived."""
    from General_Data_Analysis.Data_Pipeline import _check_archive_exists, _archive_config
    cfg = _minimal_datasets_cfg(tmp_path)
    _archive_config(cfg, _minimal_params_cfg(), 'DS')
    assert not _check_archive_exists(str(tmp_path) + '/', 'OTHER_DS')


# ---------------------------------------------------------------------------
# _archive_config
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_archive_creates_files_from_scratch(tmp_path):
    """First call creates both archived YAML files."""
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()
    cfg = _minimal_datasets_cfg(base)
    params = _minimal_params_cfg()

    _archive_config(cfg, params, 'DS')

    ds_out = yaml.safe_load(open(str(base / 'datasets.yaml')))
    p_out = yaml.safe_load(open(str(base / 'analysis_parameters.yaml')))
    assert 'DS' in ds_out['datasets']
    assert 'DS' in p_out
    assert 'paths' in ds_out
    assert 'empty_keys' in ds_out


@pytest.mark.unit
def test_archive_merges_second_dataset(tmp_path):
    """Running a second dataset appends its entry without removing the first."""
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()

    cfg_a = _minimal_datasets_cfg(base, 'DS_A')
    cfg_b = _minimal_datasets_cfg(base, 'DS_B')
    # Give each its own raw_datasets_cfg with only that dataset
    _archive_config(cfg_a, _minimal_params_cfg('DS_A'), 'DS_A')
    _archive_config(cfg_b, _minimal_params_cfg('DS_B'), 'DS_B')

    ds_out = yaml.safe_load(open(str(base / 'datasets.yaml')))
    p_out = yaml.safe_load(open(str(base / 'analysis_parameters.yaml')))
    assert 'DS_A' in ds_out['datasets']
    assert 'DS_B' in ds_out['datasets']
    assert 'DS_A' in p_out
    assert 'DS_B' in p_out


@pytest.mark.unit
def test_archive_overwrites_existing_entry(tmp_path):
    """Re-archiving the same dataset name replaces its entry."""
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()

    cfg_v1 = _minimal_datasets_cfg(base)
    cfg_v1['datasets']['DS']['pathlist'] = ['/v1/']
    _archive_config(cfg_v1, _minimal_params_cfg(), 'DS')

    cfg_v2 = _minimal_datasets_cfg(base)
    cfg_v2['datasets']['DS']['pathlist'] = ['/v2/']
    params_v2 = _minimal_params_cfg()
    params_v2['DS']['thresh'] = 999
    _archive_config(cfg_v2, params_v2, 'DS')

    ds_out = yaml.safe_load(open(str(base / 'datasets.yaml')))
    p_out = yaml.safe_load(open(str(base / 'analysis_parameters.yaml')))
    assert ds_out['datasets']['DS']['pathlist'] == ['/v2/']
    assert p_out['DS']['thresh'] == 999


@pytest.mark.unit
def test_archive_includes_aliases(tmp_path):
    """Aliases that point to the archived dataset are included."""
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()

    cfg = _minimal_datasets_cfg(base)
    cfg['aliases'] = {'my_alias': 'DS', 'unrelated_alias': 'OTHER'}
    _archive_config(cfg, _minimal_params_cfg(), 'DS')

    ds_out = yaml.safe_load(open(str(base / 'datasets.yaml')))
    assert 'my_alias' in ds_out['aliases']
    assert ds_out['aliases']['my_alias'] == 'DS'
    # alias for a dataset we didn't archive should not be present
    assert 'unrelated_alias' not in ds_out.get('aliases', {})


@pytest.mark.unit
def test_archive_idempotent(tmp_path):
    """Archiving the same dataset twice produces the same output as once."""
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()

    cfg = _minimal_datasets_cfg(base)
    params = _minimal_params_cfg()

    _archive_config(cfg, params, 'DS')
    content_after_first = open(str(base / 'datasets.yaml')).read()

    _archive_config(cfg, params, 'DS')
    content_after_second = open(str(base / 'datasets.yaml')).read()

    assert content_after_first == content_after_second


@pytest.mark.unit
def test_archive_datasets_only_when_params_none(tmp_path):
    """Passing None for analysis_params_raw skips analysis_parameters.yaml."""
    import os
    from General_Data_Analysis.Data_Pipeline import _archive_config
    base = tmp_path / 'base'
    base.mkdir()

    cfg = _minimal_datasets_cfg(base)
    _archive_config(cfg, None, 'DS')

    assert os.path.exists(str(base / 'datasets.yaml'))
    assert not os.path.exists(str(base / 'analysis_parameters.yaml'))


# ---------------------------------------------------------------------------
# Overwrite check in Data_Pipeline (using _confirm)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pipeline_aborts_on_decline(tmp_path):
    """Data_Pipeline returns early without running if user declines overwrite.

    The abort happens before load_datasets / any pipeline steps, so only file
    I/O for reading the YAML configs occurs.  We verify the archive is untouched.
    """
    import os
    from General_Data_Analysis.Data_Pipeline import _archive_config
    from General_Data_Analysis import Data_Pipeline

    base = tmp_path / 'base'
    base.mkdir()
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()

    raw_cfg = _minimal_datasets_cfg(base)
    params_raw = _minimal_params_cfg()

    datasets_yaml_path = cfg_dir / 'datasets.yaml'
    params_yaml_path = cfg_dir / 'params.yaml'
    with open(str(datasets_yaml_path), 'w') as fh:
        yaml.dump(raw_cfg, fh)
    with open(str(params_yaml_path), 'w') as fh:
        yaml.dump(params_raw, fh)

    # Pre-populate the archive so the overwrite check triggers
    _archive_config(raw_cfg, params_raw, 'DS')
    mtime_before = os.path.getmtime(str(base / 'datasets.yaml'))

    # Decline the overwrite — pipeline should return without running
    Data_Pipeline('DS',
                  datasets_yaml=str(datasets_yaml_path),
                  analysis_parameters_yaml=str(params_yaml_path),
                  _confirm=lambda msg: False)

    assert os.path.getmtime(str(base / 'datasets.yaml')) == mtime_before, \
        "Archive file must not be modified when user declines"


@pytest.mark.unit
def test_pipeline_no_prompt_when_no_archive(tmp_path, monkeypatch):
    """_confirm is never called on first run (no existing archive)."""
    import sys
    import General_Data_Analysis.Data_Pipeline_Functions  # ensure loaded
    import General_Data_Analysis.Data_Pipeline            # ensure loaded
    dpf_module = sys.modules['General_Data_Analysis.Data_Pipeline_Functions']
    dp_module   = sys.modules['General_Data_Analysis.Data_Pipeline']

    base = tmp_path / 'base'
    base.mkdir()
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()

    raw_cfg = _minimal_datasets_cfg(base)
    params_raw = _minimal_params_cfg()

    datasets_yaml_path = cfg_dir / 'datasets.yaml'
    params_yaml_path = cfg_dir / 'params.yaml'
    with open(str(datasets_yaml_path), 'w') as fh:
        yaml.dump(raw_cfg, fh)
    with open(str(params_yaml_path), 'w') as fh:
        yaml.dump(params_raw, fh)

    # _confirm should never be invoked — raise if it is
    def _should_not_be_called(msg):
        raise AssertionError(f"_confirm was called unexpectedly: {msg}")

    # Mock pipeline steps so they don't need real data files
    fake_dataset = object()
    monkeypatch.setattr(dpf_module, 'validate_dataset',
                        lambda name, ds: fake_dataset)
    fake_params = _minimal_params_cfg()['DS']
    fake_params['thresh'] = float(fake_params['thresh'])
    fake_params['bg_thresh'] = float(fake_params['bg_thresh'])
    fake_params['proj_thresh'] = float(fake_params['proj_thresh'])
    fake_params['thresh_1'] = float(fake_params['thresh_1'])
    monkeypatch.setattr(dpf_module, 'AnalysisParameters',
                        lambda p: type('P', (), fake_params)())
    for step in ('Generic_Preprocessing', 'Generic_DAQ_Preprocessing',
                 'Generic_Data_Processing', 'Generic_Image_Processing',
                 'Background_Treatment', 'filter_beams',
                 'Generic_Moment_Calculation', 'Generic_VCC_Analysis'):
        monkeypatch.setattr(dpf_module, step, lambda *a, **k: None)

    dp_module.Data_Pipeline('DS',
                             datasets_yaml=str(datasets_yaml_path),
                             analysis_parameters_yaml=str(params_yaml_path),
                             _confirm=_should_not_be_called)
