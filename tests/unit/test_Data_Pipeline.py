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
