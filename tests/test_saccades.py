import numpy as np
import pandas as pd
import pytest
from pupeyes.saccades import saccade_aoi_annulus

@pytest.fixture
def test_data():
    """Fixture providing test data for saccade tests"""
    item_coords = [(1200.0, 600.0), (800.0, 200.0), (400.0, 600.0), (800.0, 1000.0)]
    data = pd.DataFrame({
        'startx': [800, 800, 800, 800, 800, 800, 2000],  # last one has invalid start position
        'starty': [600, 600, 600, 600, 600, 600, 800],
        'endx': [1200, 400, 400, 1200, 850, 2000, 800],  # second to the last one has invalid end position
        'endy': [800, 600, 600, 600, 1020, 800, 200],
        'target_pos_el': [(1200.0, 600.0), (800.0, 1000.0), (800.0, 1000.0), (800.0, 1000.0),
                         (800.0, 200.0), (800.0, 1000.0), (800.0, 1000.0)],
        'distractor_pos_el': [(800.0, 200.0), (400.0, 600.0), (400.0, 600.0), (400.0, 600.0),
                             (800.0, 1000.0), (400.0, 600.0), (400.0, 600.0)],
        'distractor_cond': ['P', 'P', 'A', 'P', 'P', 'P', 'P'],
        'other1_pos_el': [(400.0, 600.0), (1200.0, 600.0), (1200.0, 600.0), (1200.0, 600.0),
                         (1200.0, 600.0), (1200.0, 600.0), (1200.0, 600.0)],
        'other2_pos_el': [(800.0, 1000.0), (800.0, 1000.0), (800.0, 1000.0), (800.0, 1000.0),
                         (400.0, 600.0), (800.0, 200.0), (800.0, 200.0)]
    })
    
    expected_results = {
        'normal': {
            'items': ['Target', 'Singleton', 'Non-singleton', 'Non-singleton', 'Singleton', pd.NA, pd.NA],
            'flags': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 'invalid_end_pos', 'invalid_start_pos']
        },
        'fixation': {
            'items': ['Target', 'Singleton', 'Non-singleton', 'Non-singleton', 'Singleton', pd.NA, 'Non-singleton'],
            'flags': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 'invalid_end_pos', pd.NA]
        }
    }
    
    return {
        'item_coords': item_coords,
        'data': data,
        'expected': expected_results
    }

def test_basic_classification(test_data):
    """Test basic saccade classification without fixation mode"""
    result = saccade_aoi_annulus(
        test_data['data'],
        test_data['item_coords'],
        col_startx='startx',
        col_starty='starty',
        col_endx='endx',
        col_endy='endy',
        col_target_pos='target_pos_el',
        col_distractor_pos='distractor_pos_el',
        col_distractor_cond='distractor_cond',
        col_other_pos=['other1_pos_el', 'other2_pos_el'],
        screen_dims=(1600, 1200),
        annulus_range=(50, 600),
        item_range=None,
        start_range=50,
        fixation_mode=False
    )
    
    assert result['curritem'].tolist() == test_data['expected']['normal']['items']
    assert result['flag'].tolist() == test_data['expected']['normal']['flags']

def test_classification_without_other_positions(test_data):
    """Test saccade classification without providing other positions"""
    result = saccade_aoi_annulus(
        test_data['data'],
        test_data['item_coords'],
        col_startx='startx',
        col_starty='starty',
        col_endx='endx',
        col_endy='endy',
        col_target_pos='target_pos_el',
        col_distractor_pos='distractor_pos_el',
        col_distractor_cond='distractor_cond',
        col_other_pos=None,  # not provided
        screen_dims=(1600, 1200),
        annulus_range=(50, 600),
        item_range=None,
        start_range=50,
        fixation_mode=False
    )
    
    assert result['curritem'].tolist() == test_data['expected']['normal']['items']
    assert result['flag'].tolist() == test_data['expected']['normal']['flags']

def test_fixation_mode(test_data):
    """Test saccade classification in fixation mode"""
    result = saccade_aoi_annulus(
        test_data['data'],
        test_data['item_coords'],
        col_startx='startx',
        col_starty='starty',
        col_endx='endx',
        col_endy='endy',
        col_target_pos='target_pos_el',
        col_distractor_pos='distractor_pos_el',
        col_distractor_cond='distractor_cond',
        col_other_pos=['other1_pos_el', 'other2_pos_el'],
        screen_dims=(1600, 1200),
        annulus_range=(50, 600),
        item_range=None,
        start_range=50,
        fixation_mode=True
    )
    
    assert result['curritem'].tolist() == test_data['expected']['fixation']['items']
    assert result['flag'].tolist() == test_data['expected']['fixation']['flags']

def test_invalid_inputs(test_data):
    """Test handling of invalid inputs"""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):
        saccade_aoi_annulus(
            empty_df,
            test_data['item_coords'],
            col_startx='startx',
            col_starty='starty',
            col_endx='endx',
            col_endy='endy',
            col_target_pos='target_pos_el',
            col_distractor_pos='distractor_pos_el',
            col_distractor_cond='distractor_cond',
            screen_dims=(1600, 1200)
        )
    
    # Test with invalid column names
    invalid_df = test_data['data'].copy()
    invalid_df.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']
    with pytest.raises(KeyError):
        saccade_aoi_annulus(
            invalid_df,
            test_data['item_coords'],
            col_startx='startx',
            col_starty='starty',
            col_endx='endx',
            col_endy='endy',
            col_target_pos='target_pos_el',
            col_distractor_pos='distractor_pos_el',
            col_distractor_cond='distractor_cond',
            screen_dims=(1600, 1200)
        ) 