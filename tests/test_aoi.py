from pupeyes.aoi import is_inside_singlepoint, is_inside, get_fixation_aoi, compute_aoi_statistics
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing"""
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
        'trial_data': trial_data,
        'polygon': polygon,
        'points': points
    }

def test_is_inside_singlepoint(sample_data):
    """Test point-in-polygon testing"""
    polygon = sample_data['polygon']
    points = sample_data['points']
    
    # Test single points
    assert is_inside_singlepoint(polygon, points[0]) == 1  # inside
    assert is_inside_singlepoint(polygon, points[1]) == 0  # outside
    assert is_inside_singlepoint(polygon, points[2]) == 2  # on vertex
    assert is_inside_singlepoint(polygon, points[3]) == 2  # on edge

def test_is_inside(sample_data):
    """Test parallel point-in-polygon testing"""
    polygon = sample_data['polygon']
    points = sample_data['points']
    
    results = is_inside(points, polygon)
    assert len(results) == len(points)
    assert np.all(results == [True, False, True, True])

def test_get_fixation_aoi():
    # Define test AOIs
    aois = {
        'square': [(0, 0), (0, 1), (1, 1), (1, 0)],
        'triangle': [(2, 0), (2, 2), (4, 0)]
    }
    
    # Test single point inside square
    assert get_fixation_aoi(0.5, 0.5, aois) == 'square'
    
    # Test single point inside triangle
    assert get_fixation_aoi(3, 0.5, aois) == 'triangle'
    
    # Test point outside all AOIs
    assert get_fixation_aoi(5, 5, aois) is None
    
    # Test multiple points
    x = np.array([0.5, 3, 5])
    y = np.array([0.5, 0.5, 5])
    results = get_fixation_aoi(x, y, aois)
    assert results == ['square', 'triangle', None]
    
    # Test with no AOIs
    assert get_fixation_aoi(0.5, 0.5, None) is None
    assert get_fixation_aoi(x, y, None) == [None, None, None]


def test_get_fixation_aoi2():
    # Define test AOIs
    aois = {
        'face': [(0,0), (100,0), (100,100), (0,100)],
        'text': [(150,0), (250,0), (250,50), (150,50)]
    }
    
    # Test single point inside square
    assert get_fixation_aoi(50, 50, aois) == 'face'
    
    # Test single point inside triangle
    assert get_fixation_aoi(200, 25, aois) == 'text'
    
    # Test point outside all AOIs
    assert get_fixation_aoi(300, 300, aois) is None
    
    # Test multiple points
    x = np.array([50, 200, 300])
    y = np.array([50, 25, 300])
    results = get_fixation_aoi(x, y, aois)
    assert results == ['face', 'text', None]
    
    # Test with no AOIs
    assert get_fixation_aoi(50, 50, None) is None
    assert get_fixation_aoi(x, y, None) == [None, None, None]

def test_compute_aoi_statistics():
    # Define test AOIs
    aois = {
        'square': [(0, 0), (0, 1), (1, 1), (1, 0)],
        'triangle': [(2, 0), (2, 2), (4, 0)]
    }
    
    # Test points and durations
    x = np.array([0.5, 3, 5, 0.2])
    y = np.array([0.5, 0.5, 5, 0.3])
    durations = np.array([100, 200, 300, 400])
    
    stats = compute_aoi_statistics(x, y, aois, durations)
    
    # Check structure and values
    assert set(stats.keys()) == {'outside', 'square', 'triangle'}
    assert stats['square']['count'] == 2
    assert stats['square']['total_duration'] == 500  # 100 + 400
    assert stats['triangle']['count'] == 1
    assert stats['triangle']['total_duration'] == 200
    assert stats['outside']['count'] == 1
    assert stats['outside']['total_duration'] == 300
    
    # Test with empty AOIs
    empty_stats = compute_aoi_statistics(x, y, {})
    assert empty_stats == {}
    
    # Test without durations
    stats_no_duration = compute_aoi_statistics(x, y, aois)
    assert stats_no_duration['square']['count'] == 2
    assert stats_no_duration['triangle']['count'] == 1
    assert stats_no_duration['outside']['count'] == 1

def test_compute_aoi_statistics2():
    # Define test AOIs
    aois = {
        'face': [(0,0), (100,0), (100,100), (0,100)],
        'text': [(150,0), (250,0), (250,50), (150,50)]
    }
    
    # Test points and durations
    x = np.array([50, 200, 300]) # face, text, outside
    y = np.array([50, 25, 300]) 
    durations = np.array([100, 200, 300])
    
    stats = compute_aoi_statistics(x, y, aois, durations)
    
    # Check structure and values
    assert set(stats.keys()) == {'outside', 'face', 'text'}
    assert stats['face']['count'] == 1
    assert stats['face']['total_duration'] == 100
    assert stats['text']['count'] == 1
    assert stats['text']['total_duration'] == 200
    assert stats['outside']['count'] == 1
    assert stats['outside']['total_duration'] == 300
    
    # Test with empty AOIs
    empty_stats = compute_aoi_statistics(x, y, {})
    assert empty_stats == {}
    
    # Test without durations
    stats_no_duration = compute_aoi_statistics(x, y, aois)
    assert stats_no_duration['face']['count'] == 1
    assert stats_no_duration['text']['count'] == 1
    assert stats_no_duration['outside']['count'] == 1


if __name__ == '__main__':
    pytest.main([__file__]) 