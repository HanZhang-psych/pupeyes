import numpy as np
import pytest
from ..aoi import aoi_stats_parallel, get_fixation_aoi, compute_aoi_statistics

def test_aoi_stats_parallel():
    # Test case 1: Basic functionality
    aoi_assignments = np.array([-1, 0, 1, 0, -1])  # 2 AOIs + outside
    durations = np.array([100, 200, 300, 400, 500])
    n_aois = 2
    
    counts, total_durations = aoi_stats_parallel(aoi_assignments, n_aois, durations)
    
    assert len(counts) == n_aois + 1  # +1 for outside
    assert len(total_durations) == n_aois + 1
    assert counts[0] == 2  # outside count
    assert counts[1] == 2  # AOI 0 count
    assert counts[2] == 1  # AOI 1 count
    assert total_durations[0] == 600  # outside duration
    assert total_durations[1] == 600  # AOI 0 duration
    assert total_durations[2] == 300  # AOI 1 duration

    # Test case 2: No durations provided
    counts, total_durations = aoi_stats_parallel(aoi_assignments, n_aois)
    assert np.all(total_durations == 0)
    assert np.array_equal(counts, np.array([2, 2, 1]))

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

if __name__ == '__main__':
    pytest.main([__file__]) 