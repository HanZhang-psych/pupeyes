import pytest
import numpy as np
import pandas as pd
import scipy.signal as signal
from ..utils import (
    lowpass_filter, make_mask, convert_coordinates, get_isoeccentric_positions,
    xy_circle, xy_from_polar, is_inside, is_inside_parallel, angular_distance,
    gaussian_2d, mat2gray
)

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing"""
    # Create sample signal data
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    # Create sample trial data
    trial_data = pd.DataFrame({
        'trial': [1, 1, 2, 2, 3, 3],
        'condition': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    
    # Create sample polygon data
    polygon = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    points = np.array([
        (0.5, 0.5),  # inside
        (2, 2),      # outside
        (0, 0),      # on vertex
        (0.5, 0)     # on edge
    ])
    
    return {
        'signal': signal,
        'time': t,
        'trial_data': trial_data,
        'polygon': polygon,
        'points': points
    }

def test_lowpass_filter(sample_data):
    """Test lowpass filter functionality"""
    signal = sample_data['signal']
    
    # Test basic filtering
    filtered = lowpass_filter(signal, sampling_freq=1000, cutoff_freq=20)
    assert len(filtered) == len(signal)
    assert np.all(np.abs(filtered) <= np.abs(signal).max())
    
    # Test different parameters
    filtered_higher = lowpass_filter(signal, sampling_freq=1000, cutoff_freq=40)
    filtered_lower = lowpass_filter(signal, sampling_freq=1000, cutoff_freq=5)
    assert np.mean(np.abs(filtered_higher)) > np.mean(np.abs(filtered_lower))
    
    # Test with different orders
    filtered_order2 = lowpass_filter(signal, sampling_freq=1000, cutoff_freq=20, order=2)
    filtered_order4 = lowpass_filter(signal, sampling_freq=1000, cutoff_freq=20, order=4)
    assert len(filtered_order2) == len(filtered_order4)

def test_make_mask(sample_data):
    """Test mask creation functionality"""
    data = sample_data['trial_data']
    
    # Test with dictionary input
    mask_dict = {'trial': 1, 'condition': 'A'}
    mask = make_mask(data, mask_dict)
    assert len(mask) == len(data)
    assert mask.sum() == 4  # Should mask out 2 rows
    
    # Test with DataFrame input
    mask_df = pd.DataFrame({'trial': [1], 'condition': ['A']})
    mask_df_result = make_mask(data, mask_df)
    assert np.array_equal(mask, mask_df_result)
    
    # Test invert parameter
    mask_inverted = make_mask(data, mask_dict, invert=True)
    assert np.array_equal(mask, ~mask_inverted)

def test_convert_coordinates():
    """Test coordinate conversion functionality"""
    # Test psychopy to eyelink conversion
    psychopy_coord = [0, 0]  # center
    el_coord = convert_coordinates(psychopy_coord, screen_dims=[1600, 1200])
    assert el_coord[0] == 800  # half width
    assert el_coord[1] == 600  # half height
    
    # Test eyelink to psychopy conversion
    el_coord = [800, 600]  # center in eyelink coords
    psychopy_coord = convert_coordinates(el_coord, direction='to_psychopy')
    assert psychopy_coord[0] == 0
    assert psychopy_coord[1] == 0
    
    # Test string input
    str_coord = "100,100"
    el_coord = convert_coordinates(str_coord)
    assert isinstance(el_coord, np.ndarray)
    assert el_coord[0] == 900
    assert el_coord[1] == 500

    # Test different units
    norm_coord = [0.5, 0.5]
    pix_coord = convert_coordinates(norm_coord, psychopy_units='norm')
    assert pix_coord[0] > norm_coord[0]
    assert pix_coord[0] == 1200
    assert pix_coord[1] == 300

    height_coord = [0.5, 0.5] # 50% of screen height
    pix_coord = convert_coordinates(height_coord, screen_dims=[1600, 1200], psychopy_units='height')
    assert pix_coord[0] == 1400
    assert pix_coord[1] == 0


def test_get_isoeccentric_positions():
    """Test isoeccentric position generation"""
    # Test basic functionality
    positions = get_isoeccentric_positions(4, 100)
    assert len(positions) == 4
    assert np.allclose(positions, np.array([(100, 0), (0, 100), (-100, 0), (0, -100)]))

    # Test with offset
    positions_offset = get_isoeccentric_positions(4, 100, offset_deg=45, round_to=0)
    assert len(positions_offset) == 4
    assert positions != positions_offset
    assert np.allclose(positions_offset, np.array([(71, 71), (-71, 71), (-71, -71), (71, -71)]))

    # Test different coordinate systems
    pos_psychopy = get_isoeccentric_positions(4, 100, coordinate_system='psychopy')
    pos_eyelink = get_isoeccentric_positions(4, 100, coordinate_system='eyelink')
    assert pos_psychopy != pos_eyelink
    assert np.allclose(pos_psychopy, np.array([(100, 0), (0, 100), (-100, 0), (0, -100)]))
    assert np.allclose(pos_eyelink, np.array([(900, 600), (800, 500), (700, 600), (800, 700)]))

def test_xy_circle():
    """Test circular coordinate generation"""
    # Test basic functionality
    points = xy_circle(4, 100)
    assert len(points) == 4
    assert np.allclose(points, np.array([(100, 0), (0, 100), (-100, 0), (0, -100)]))
    
    # Test with rotation
    points_rotated = xy_circle(4, 100, phi0=45)
    assert points != points_rotated
    assert np.allclose(points_rotated, np.array([(70.71, 70.71), (-70.71, 70.71), (-70.71, -70.71), (70.71, -70.71)]))
    
    # Test with different pole
    points_offset = xy_circle(4, 100, pole=(50, 50))
    assert points != points_offset
    assert np.allclose(points_offset, np.array([(150, 50), (50, 150), (-50, 50), (50, -50)]))

def test_xy_from_polar():
    """Test polar to cartesian conversion"""
    # Test basic conversion
    x, y = xy_from_polar(100, 0)
    assert np.isclose(x, 100)
    assert np.isclose(y, 0)
    
    # Test 45 degree angle
    x, y = xy_from_polar(100, 45)
    assert np.isclose(x, 100/np.sqrt(2))
    assert np.isclose(y, 100/np.sqrt(2))
    
    # Test with pole offset
    x, y = xy_from_polar(100, 0, pole=(50, 50))
    assert np.isclose(x, 150)
    assert np.isclose(y, 50)

def test_is_inside(sample_data):
    """Test point-in-polygon testing"""
    polygon = sample_data['polygon']
    points = sample_data['points']
    
    # Test single points
    assert is_inside(polygon, points[0]) == 1  # inside
    assert is_inside(polygon, points[1]) == 0  # outside
    assert is_inside(polygon, points[2]) == 2  # on vertex
    assert is_inside(polygon, points[3]) == 2  # on edge

def test_is_inside_parallel(sample_data):
    """Test parallel point-in-polygon testing"""
    polygon = sample_data['polygon']
    points = sample_data['points']
    
    results = is_inside_parallel(points, polygon)
    assert len(results) == len(points)
    assert results[0]  # inside
    assert not results[1]  # outside

def test_angular_distance():
    """Test angular distance calculation"""
    # Test parallel lines
    line1 = [(0, 0), (1, 0)]
    line2 = [(0, 1), (1, 1)]
    assert np.isclose(angular_distance(line1, line2), 0)
    
    # Test perpendicular lines
    line2 = [(0, 0), (0, 1)]
    assert np.isclose(angular_distance(line1, line2), 90)
    
    # Test 45 degree angle
    line2 = [(0, 0), (1, 1)]
    assert np.isclose(angular_distance(line1, line2), 45)

def test_gaussian_2d():
    """Test 2D Gaussian filtering"""
    # Create test image
    img = np.zeros((100, 100))
    img[40:60, 40:60] = 1
    
    # Test filtering
    filtered = gaussian_2d(img, fc=10)
    assert filtered.shape == img.shape
    
    # Due to FFT computation, very small negative values might occur
    # Check that any negative values are negligibly small
    min_val = filtered.min()
    if min_val < 0:
        assert abs(min_val) < 1e-10, f"Negative values too large: {min_val}"
    
    # Check maximum value is reasonable
    assert filtered.max() <= 1.0 + 1e-10
    
    # Test that the filter preserves total energy approximately
    # The sum should be similar to the original, within some tolerance
    assert np.abs(np.sum(filtered) - np.sum(img)) < np.sum(img) * 0.1
    
    # Test that the filter actually smooths the image
    # The maximum value should be less than the original due to smoothing
    assert filtered.max() < img.max()
    
    # Test different cutoff frequencies
    filtered_high = gaussian_2d(img, fc=20)
    filtered_low = gaussian_2d(img, fc=5)
    
    # Higher cutoff frequency should preserve more of the original image structure
    # Calculate the correlation with the original image
    corr_high = np.corrcoef(img.flatten(), filtered_high.flatten())[0, 1]
    corr_low = np.corrcoef(img.flatten(), filtered_low.flatten())[0, 1]
    assert corr_high > corr_low, "Higher cutoff frequency should better preserve original image structure"

def test_mat2gray():
    """Test grayscale conversion"""
    # Test with simple array
    img = np.array([[0, 128, 255], [64, 192, 255]], dtype=np.uint8)
    gray = mat2gray(img)
    assert gray.min() >= 0
    assert gray.max() <= 1
    
    # Test with float array
    img_float = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 1.0]])
    gray_float = mat2gray(img_float)
    assert gray_float.min() >= 0
    assert gray_float.max() <= 1

if __name__ == '__main__':
    pytest.main([__file__]) 