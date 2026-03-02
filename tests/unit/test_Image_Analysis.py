import numpy as np
import pytest
from General_Data_Analysis.Image_Analysis import GaussianParams


@pytest.mark.unit
def test_gaussian_params_init():
    params_dict = {"Amplitude": 2.0, "Center": 0.0, "Sigma": 1.0, "Baseline": 0.5}
    params = GaussianParams(params_dict)
	
    assert params.Amplitude == 2.0
    assert params.Center == 0.0
    assert params.Sigma == 1.0
    assert params.Baseline == 0.5

@pytest.mark.unit
def test_gaussian_evaluate_known():
    params_dict = {"Amplitude": 1.0, "Center": 0.0, "Sigma": 1.0, "Baseline": 0.0}
    params = GaussianParams(params_dict)
    x = np.array([0.0, 0.0])
    print(len(np.shape(x)))
    expected = np.array([1.0, 1.0])
    result = params.evaluate_gaussian(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)

	# Test at x = 1, should be exp(-0.5)
    x = np.array([1.0, 1.0])
    print(len(np.shape(x)))
    expected = np.array([np.exp(-0.5), np.exp(-0.5)])
    result = params.evaluate_gaussian(x)
    np.testing.assert_allclose(result, expected, rtol=1e-7)
@pytest.mark.unit
def test_eval_gauss_baseline_math():
    from General_Data_Analysis.Image_Analysis import eval_gauss_baseline
    import numpy as np

    # Example 1: amplitude=1, center=0, sigma=1, baseline=0, x=0
    result = eval_gauss_baseline([0.0, 1.0, 0.0, 1.0], [0.0, 0.0])
    expected = np.array([1.0, 1.0])  # exp(0) = 1
    np.testing.assert_allclose(result, expected, rtol=1e-7)

    # Example 2: amplitude=2, center=1, sigma=2, baseline=0.5, x=1
    result = eval_gauss_baseline([1.0, 2.0, 2.0, 0.5], [1.0, 1.0])
    expected = np.array([2.0 + 0.5, 2.0 + 0.5])  # exp(0) = 1
    np.testing.assert_allclose(result, expected, rtol=1e-7)

    # Example 3: amplitude=3, center=0, sigma=1, baseline=1, x=1
    result = eval_gauss_baseline([0.0, 1.0, 3.0, 1.0], [1.0, 1.0])
    expected = np.array([3.0 * np.exp(-0.5) + 1.0, 3.0 * np.exp(-0.5) + 1.0])
    np.testing.assert_allclose(result, expected, rtol=1e-7)

import numpy as np
import pytest
from General_Data_Analysis.Image_Analysis import penalty_func

@pytest.mark.unit
def test_penalty_func_exact_match():
    # Gaussian: center=0, sigma=1, amplitude=1, baseline=0
    p = np.array([0.0, 1.0, 1.0, 0.0])
    x = np.array([0.0, 1.0])
    v = 1.0 * np.exp(-0.5 * (x - 0.0) ** 2 / (1.0 ** 2)) + 0.0
    # penalty should be zero for perfect match
    assert penalty_func(p, v, x) == pytest.approx(0.0)

@pytest.mark.unit
def test_penalty_func_offset():
    # Gaussian: center=0, sigma=1, amplitude=1, baseline=0
    p = np.array([0.0, 1.0, 1.0, 0.0])
    x = np.array([0.0, 1.0])
    v = np.array([2.0, 2.0])  # offset by +1 everywhere
    # penalty should be sum of squared differences: (1-2)^2 + (exp(-0.5)-2)^2
    expected = np.sum((1.0 * np.exp(-0.5 * (x - 0.0) ** 2 / (1.0 ** 2)) - v) ** 2)
    assert penalty_func(p, v, x) == pytest.approx(expected)

@pytest.mark.unit
def test_penalty_func_known_values():
    # Gaussian: center=1, sigma=2, amplitude=2, baseline=0.5
    p = np.array([1.0, 2.0, 2.0, 0.5])
    x = np.array([1.0, 3.0])
    v = np.array([2.5, 0.5])  # at x=1: 2+0.5=2.5, at x=3: 2*exp(-0.5)=2*exp(-0.5*4/4)=2*exp(-0.5)=~1.213 + 0.5
    expected = ((2.5 - 2.5) ** 2 + ((2.0 * np.exp(-0.5) + 0.5) - 0.5) ** 2)
    assert penalty_func(p, v, x) == pytest.approx(expected)


import numpy as np
from General_Data_Analysis.Image_Analysis import init_guess, GaussianParams
@pytest.mark.unit
def test_init_guess_basic():
    # Create a simple projection: peak at index 5, baseline 2, amplitude 10
    xproj = np.ones(20) * 2
    xproj[5] = 12  # peak at index 5
    lengthscale = 10

    guess = init_guess(xproj, lengthscale)
    # Baseline should be mean of first 10 elements (all 2 except index 5)
    expected_base = np.mean(xproj[:10])
    # Center should be index of max (5)
    expected_center = 5
    # Amplitude should be max - baseline (12 - expected_base)
    expected_amplitude = 12 - expected_base
    # Sigma should be len(xproj)/lengthscale
    expected_sigma = len(xproj) / lengthscale

    assert isinstance(guess, GaussianParams)
    assert guess.Center == expected_center
    assert np.isclose(guess.Baseline, expected_base)
    assert np.isclose(guess.Amplitude, expected_amplitude)
    assert np.isclose(guess.Sigma, expected_sigma)
@pytest.mark.unit
def test_init_guess_lengthscale_type():
    xproj = np.arange(20)
    guess = init_guess(xproj, lengthscale="10")
    assert isinstance(guess, GaussianParams)
    assert np.isclose(guess.Sigma, 2.0)


import numpy as np
from General_Data_Analysis.Image_Analysis import fit_gauss, GaussianParams, eval_gauss_baseline
@pytest.mark.unit
def test_fit_gauss_perfect_gaussian():
    # True parameters: center=10, sigma=3, amplitude=5, baseline=2
    true_params = np.array([10.0, 3.0, 5.0, 2.0])
    x = np.arange(0, 20, 1)
    y = eval_gauss_baseline(true_params, x)

    # Fit the gaussian
    fit = fit_gauss(y)

    # Check that the fitted parameters are close to the true parameters
    assert np.isclose(fit.Center, true_params[0], atol=1e-1)
    assert np.isclose(fit.Sigma, true_params[1], atol=1e-1)
    assert np.isclose(fit.Amplitude, true_params[2], atol=1e-1)
    assert np.isclose(fit.Baseline, true_params[3], atol=1e-1)
@pytest.mark.unit
def test_fit_gauss_with_noise():
    # True parameters
    true_params = np.array([8.0, 2.0, 4.0, 1.0])
    x = np.arange(0, 16, 1)
    y = eval_gauss_baseline(true_params, x)
    # Add small noise
    y_noisy = y + np.random.normal(0, 0.05, size=y.shape)

    fit = fit_gauss(y_noisy)

    assert np.isclose(fit.Center, true_params[0], atol=0.2)
    assert np.isclose(fit.Sigma, true_params[1], atol=0.2)
    assert np.isclose(fit.Amplitude, true_params[2], atol=0.2)
    assert np.isclose(fit.Baseline, true_params[3], atol=0.2)



import numpy as np
from General_Data_Analysis.Image_Analysis import imrotate45
@pytest.mark.unit
def test_imrotate45_shape_and_fill():
    # Create a simple 5x5 image with a central value
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255
    bg = 7

    # Rotate the image
    rotated = imrotate45(img, bg)

    # After 45 degree rotation, the bounding box should be larger
    assert rotated.shape[0] > img.shape[0]
    assert rotated.shape[1] > img.shape[1]
    # Check that the new shape is approximately sqrt(2) times larger
    expected_size = int(np.ceil(img.shape[0] * np.sqrt(2)))
    assert abs(rotated.shape[0] - expected_size) <= 1
    assert abs(rotated.shape[1] - expected_size) <= 1

    # Check that the fill color is present in the corners
    corners = [
        rotated[0, 0],
        rotated[0, -1],
        rotated[-1, 0],
        rotated[-1, -1]
    ]
    for val in corners:
        assert val == bg
@pytest.mark.unit
def test_imrotate45_preserves_center():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255
    bg = 0

    rotated = imrotate45(img, bg)
    # The maximum value should still be present after rotation
    assert np.max(rotated) == 255
@pytest.mark.unit
def test_imrotate45_invalid_bg():
    img = np.zeros((5, 5), dtype=np.uint8)
    try:
        imrotate45(img, "not_an_int")
        assert False, "Should raise ValueError for non-integer bg"
    except ValueError:
        pass
@pytest.mark.unit
def test_imrotate45_counterclockwise_rotation():
    # Create a simple asymmetric image: a vertical line at column 1
    img = np.zeros((7, 7), dtype=np.uint8)
    img[:, 1] = 255
    bg = 0

    # Rotate the image
    rotated = imrotate45(img, bg)

    # After 45 degree CCW rotation, the vertical line should become a diagonal line
    # Find the coordinates of nonzero pixels in the rotated image
    coords = np.argwhere(rotated == 255)

    # The coordinates should roughly follow a diagonal (y = x or y = x + c)
    # We'll check that the spread of (row - col) is small
    diffs = coords[:, 0] - coords[:, 1]
    # For a perfect diagonal, all diffs would be equal
    assert np.std(diffs) < 2, "Rotated line should be diagonal after 45 degree CCW rotation"

import numpy as np
from General_Data_Analysis.Image_Analysis import Gaussian_Fit_4_Dim, GaussianParams, eval_gauss_baseline, imrotate45


def make_2d_gaussian(shape, center, sigma, amplitude=1.0, baseline=0.0):
    """Create a 2D Gaussian image."""
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    xx, yy = np.meshgrid(x, y)
    gauss = amplitude * np.exp(-((xx - center[1]) ** 2 + (yy - center[0]) ** 2) / (2 * sigma ** 2)) + baseline
    return gauss
@pytest.mark.unit
def test_Gaussian_Fit_4_Dim_rotated_gaussian():
    # Create a 2D Gaussian image
    shape = (32, 32)
    center = (16, 16)
    sigma = 4.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    # Fit projections and rotated projections
    xfit, yfit, xfit45, yfit45 = Gaussian_Fit_4_Dim(img)

    # Check that the fitted parameters for x and y are close to the true values
    assert np.isclose(xfit.Center, center[1], atol=1.0)
    assert np.isclose(yfit.Center, center[0], atol=1.0)
    assert np.isclose(xfit.Sigma, sigma, atol=1.0)
    assert np.isclose(yfit.Sigma, sigma, atol=1.0)
    # assert np.isclose(xfit.Amplitude, amplitude, atol=1.0)
    # assert np.isclose(yfit.Amplitude, amplitude, atol=1.0)
    assert np.isclose(xfit.Baseline / shape[1], baseline, atol=1.0)
    assert np.isclose(yfit.Baseline / shape[0], baseline, atol=1.0)

    # For rotated projections, check that the amplitude and baseline are reasonable
    assert xfit45.Amplitude > 0
    assert yfit45.Amplitude > 0
    assert xfit45.Baseline >= 0
    assert yfit45.Baseline >= 0
    # Centers should be near the middle of the rotated image
    rotated_img = imrotate45(img, baseline)
    mid_x = rotated_img.shape[1] // 2
    mid_y = rotated_img.shape[0] // 2
    assert abs(xfit45.Center - mid_x) < 5
    assert abs(yfit45.Center - mid_y) < 5


import numpy as np
from General_Data_Analysis.Image_Analysis import bg_thresh
@pytest.mark.unit
def test_bg_thresh_no_outliers():
    # All values are the same, std should be zero
    bg = np.ones((10, 10)) * 5
    std = bg_thresh(bg)
    assert np.isclose(std, 0.0)

@pytest.mark.unit
def test_bg_thresh_with_outliers():
    # Most values are 5, a few are 100 (outliers)
    bg = np.ones((10, 10)) * 5
    bg[0, 0] = 100
    bg[1, 1] = 100
    std = bg_thresh(bg)
    # Outliers should be removed, std should be close to zero
    assert std < 1e-6
@pytest.mark.unit
def test_bg_thresh_typical_spread():
    # Normal distributed values, std should be close to 1
    rng = np.random.default_rng(42)
    bg = rng.normal(0, 1, (100, 100))
    std = bg_thresh(bg)
    assert np.isclose(std, 1.0, atol=0.1)


import numpy as np
from General_Data_Analysis.Image_Analysis import RMS_Calc, GaussianParams
@pytest.mark.unit
def test_RMS_Calc_perfect_gaussian():
    # Create a perfect 1D Gaussian
    center = 10
    sigma = 2.5
    amplitude = 5.0
    baseline = 0.0
    x = np.arange(20)
    data = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + baseline

    params = RMS_Calc(data)
    assert isinstance(params, GaussianParams)
    # Center should be close to true center
    assert np.isclose(params.Center, center, atol=0.1)
    # Sigma should be close to true sigma
    assert np.isclose(params.Sigma, sigma, atol=0.1)
    # Amplitude should be close to peak value
    assert np.isclose(params.Amplitude, amplitude, atol=0.1)
    # Baseline should be zero
    assert np.isclose(params.Baseline, 0.0)

@pytest.mark.unit
def test_RMS_Calc_uniform():
    # Uniform distribution, center should be at midpoint, sigma should be known
    data = np.ones(10)
    params = RMS_Calc(data)
    assert np.isclose(params.Center, 4.5, atol=0.1)
    # For uniform [0,9], sigma = sqrt(sum((x-4.5)^2)/10)
    x = np.arange(10)
    expected_sigma = np.sqrt(np.mean((x - 4.5) ** 2))
    assert np.isclose(params.Sigma, expected_sigma, atol=0.1)

from General_Data_Analysis.Image_Analysis import RMS_Image_Analysis, GaussianParams

@pytest.mark.unit
def test_RMS_Image_Analysis_rotated_gaussian():
    # Create a 2D Gaussian image
    shape = (32, 32)
    center = (16, 16)
    sigma = 4.0
    amplitude = 10.0
    baseline = 0.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    # Fit projections and rotated projections
    xfit, yfit, xfit45, yfit45 = RMS_Image_Analysis(img)

    # Check that the fitted parameters for x and y are close to the true values
    assert np.isclose(xfit.Center, center[1], atol=1.0)
    assert np.isclose(yfit.Center, center[0], atol=1.0)
    assert np.isclose(xfit.Sigma, sigma, atol=1.0)
    assert np.isclose(yfit.Sigma, sigma, atol=1.0)
    # assert np.isclose(xfit.Amplitude, amplitude + baseline, atol=1.0)
    # assert np.isclose(yfit.Amplitude, amplitude + baseline, atol=1.0)
    assert np.isclose(xfit.Baseline, 0.0, atol=1e-6)
    assert np.isclose(yfit.Baseline, 0.0, atol=1e-6)

    # For rotated projections, check that the amplitude is reasonable
    assert xfit45.Amplitude > 0
    assert yfit45.Amplitude > 0
    # Centers should be near the middle of the rotated image
    mid_x = xfit45.Center
    mid_y = yfit45.Center
    rotated_shape = (32 * np.sqrt(2), 32 * np.sqrt(2))  # Approximate expanded shape
    assert abs(mid_x - rotated_shape[1] / 2) < 10
    assert abs(mid_y - rotated_shape[0] / 2) < 10


from General_Data_Analysis.Image_Analysis import image_cropp_center, GaussianParams
@pytest.mark.unit
def test_image_cropp_center_basic_crop():
    # Create a 2D Gaussian image
    shape = (40, 40)
    center = (20, 20)
    sigma = 5.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    # Provide fits directly
    xfit = GaussianParams([center[1], amplitude + baseline, sigma, baseline])
    yfit = GaussianParams([center[0], amplitude + baseline, sigma, baseline])

    num_std = 2
    cropped = image_cropp_center(img, num_std, xfit=xfit, yfit=yfit)

    # Cropped shape should be about 2*sigma*2 in each direction
    expected_shape = (int(np.ceil(num_std * sigma) * 2), int(np.ceil(num_std * sigma) * 2))
    assert cropped.shape[0] <= expected_shape[0] + 2  # allow for rounding
    assert cropped.shape[1] <= expected_shape[1] + 2
@pytest.mark.unit
def test_image_cropp_center_centered_peak():
    shape = (40, 40)
    center = (20, 20)
    sigma = 5.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    cropped = image_cropp_center(img, 2)
    # The peak should be near the center of the cropped image
    peak_pos = np.unravel_index(np.argmax(cropped), cropped.shape)
    center_pos = (cropped.shape[0] // 2, cropped.shape[1] // 2)
    assert abs(peak_pos[0] - center_pos[0]) <= 2
    assert abs(peak_pos[1] - center_pos[1]) <= 2
@pytest.mark.unit
def test_image_cropp_center_edge_case():
    # Gaussian near the edge
    shape = (40, 40)
    center = (5, 5)
    sigma = 3.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)
    cropped = image_cropp_center(img, 2)
    # Should not crop outside image bounds
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0

from General_Data_Analysis.Image_Analysis import ellipse_crop_v3, GaussianParams

@pytest.mark.unit
def test_ellipse_crop_v3_basic_masking():
    # Create a 2D Gaussian image
    shape = (40, 40)
    center = (20, 20)
    sigma = 5.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    # Crop with ellipse_crop_v3
    cropped = ellipse_crop_v3(img, sigmaThresh=2)

    # Check that the cropped image has zeros outside the ellipse
    assert np.any(cropped == 0)
    # The center should still have nonzero values
    assert cropped[int(center[0]), int(center[1])] > 0
@pytest.mark.unit
def test_ellipse_crop_v3_return_bg():
    shape = (40, 40)
    center = (20, 20)
    sigma = 5.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian(shape, center, sigma, amplitude, baseline)

    cropped, bg = ellipse_crop_v3(img, sigmaThresh=2, return_bg=True)
    # bg should be all pixels outside the ellipse, and should be mostly baseline
    assert np.all(bg >= baseline)
    # assert np.any(bg == baseline)

def make_2d_gaussian_aniso(shape, center, sigma_x, sigma_y, amplitude=1.0, baseline=0.0):
    """Create a 2D Gaussian image with different sigma in x and y."""
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    xx, yy = np.meshgrid(x, y)
    gauss = amplitude * np.exp(-((xx - center[1]) ** 2 / (2 * sigma_x ** 2) + (yy - center[0]) ** 2 / (2 * sigma_y ** 2))) + baseline
    return gauss

@pytest.mark.unit
def test_ellipse_crop_v3_anisotropic_gaussian():
    # Create a 2D anisotropic Gaussian image
    shape = (40, 40)
    center = (20, 20)
    sigma_x = 8.0
    sigma_y = 3.0
    amplitude = 10.0
    baseline = 2.0
    img = make_2d_gaussian_aniso(shape, center, sigma_x, sigma_y, amplitude, baseline)

    # Crop with ellipse_crop_v3
    cropped = ellipse_crop_v3(img, sigmaThresh=2)

    # Check that the cropped image has zeros outside the ellipse
    assert np.any(cropped == 0)
    # The center should still have nonzero values
    assert cropped[int(center[0]), int(center[1])] > 0

    # Check that the spread in x is greater than in y
    xproj = np.sum(cropped, axis=0)
    yproj = np.sum(cropped, axis=1)
    x_width = np.sum(xproj > baseline)
    y_width = np.sum(yproj > baseline)
    assert x_width > y_width