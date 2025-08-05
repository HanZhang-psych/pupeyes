#!/usr/bin/env python3
"""
Test script to verify PupEyes Docker setup
Run this inside the Docker container to check if everything is working
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    packages = [
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'plotly',
        'dash',
        'opencv-python',
        'h5py',
        'tables',
        'tqdm',
        'intervaltree',
        'dill',
        'ipywidgets',
        'nbformat'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            # Handle special cases
            if package == 'opencv-python':
                importlib.import_module('cv2')
            else:
                importlib.import_module(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_pupeyes():
    """Test if PupEyes can be imported and basic functionality works"""
    print("\nTesting PupEyes imports...")
    
    pupeyes_modules = [
        'pupeyes',
        'pupeyes.pupil',
        'pupeyes.aoi',
        'pupeyes.saccades',
        'pupeyes.data.eyelink',
        'pupeyes.data.tobii_titta',
        'pupeyes.apps.pupil_viewer',
        'pupeyes.apps.fixation_viewer',
        'pupeyes.apps.aoi_drawer'
    ]
    
    failed_imports = []
    
    for module in pupeyes_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_jupyter():
    """Test if Jupyter is working"""
    print("\nTesting Jupyter...")
    
    try:
        import jupyter
        import notebook
        import jupyterlab
        print("✓ Jupyter packages available")
        return []
    except ImportError as e:
        print(f"✗ Jupyter: {e}")
        return ['jupyter']

def test_data_access():
    """Test if sample data is accessible"""
    print("\nTesting data access...")
    
    import os
    
    data_paths = [
        'docs/data',
        'docs/data/sub001.asc',
        'docs/data/sub002.asc',
        'docs/data/sub003.asc',
        'docs/data/sub004.asc'
    ]
    
    failed_paths = []
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} (not found)")
            failed_paths.append(path)
    
    return failed_paths

def main():
    """Run all tests"""
    print("=" * 50)
    print("PupEyes Docker Test Suite")
    print("=" * 50)
    
    # Test imports
    failed_packages = test_imports()
    
    # Test PupEyes
    failed_pupeyes = test_pupeyes()
    
    # Test Jupyter
    failed_jupyter = test_jupyter()
    
    # Test data access
    failed_data = test_data_access()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_failures = len(failed_packages) + len(failed_pupeyes) + len(failed_jupyter) + len(failed_data)
    
    if total_failures == 0:
        print("🎉 All tests passed! PupEyes Docker setup is working correctly.")
        print("\nYou can now:")
        print("- Access Jupyter Lab at http://localhost:8888")
        print("- Run the notebooks in the docs/ directory")
        print("- Use the interactive applications")
    else:
        print(f"❌ {total_failures} test(s) failed:")
        
        if failed_packages:
            print(f"  - {len(failed_packages)} package import(s) failed")
        if failed_pupeyes:
            print(f"  - {len(failed_pupeyes)} PupEyes module(s) failed")
        if failed_jupyter:
            print(f"  - {len(failed_jupyter)} Jupyter component(s) failed")
        if failed_data:
            print(f"  - {len(failed_data)} data file(s) not found")
        
        print("\nPlease check the Docker setup and rebuild if necessary.")
    
    return total_failures == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 