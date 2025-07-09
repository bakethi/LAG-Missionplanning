#!/usr/bin/env python3
"""
Usage script for advanced multiwaypoint mission analysis

This script demonstrates how to:
1. Run the multiwaypoint mission with advanced data collection
2. Analyze the results with MATLAB
3. Generate comprehensive reports

Requirements:
- Python: matplotlib, scipy, numpy, torch
- MATLAB: with access to the analyze_multiwaypoint_mission.m function
"""

import os
import subprocess
import sys
from pathlib import Path

def run_multiwaypoint_analysis():
    """Run the complete multiwaypoint analysis pipeline"""
    
    print("="*60)
    print("MULTIWAYPOINT MISSION ADVANCED ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Run the advanced multiwaypoint rendering
    print("\n1. Running multiwaypoint mission with advanced data collection...")
    try:
        result = subprocess.run([sys.executable, "render_multiwaypoint_advanced.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error running multiwaypoint mission: {result.stderr}")
            return False
        print("✓ Multiwaypoint mission completed successfully")
    except subprocess.TimeoutExpired:
        print("✗ Multiwaypoint mission timed out")
        return False
    except Exception as e:
        print(f"✗ Error running multiwaypoint mission: {e}")
        return False
    
    # Step 2: Find the generated .mat file
    print("\n2. Locating generated data files...")
    
    # Look for the most recent MultiWaypointEnv directory in current and renders folders
    search_paths = [Path("."), Path("../renders")]
    multiwaypoint_dirs = []
    
    for search_path in search_paths:
        if search_path.exists():
            # Look for MultiWaypointEnv directories
            parent_dirs = [d for d in search_path.iterdir() 
                          if d.is_dir() and d.name.startswith("MultiWaypointEnv")]
            
            # Look for timestamped subdirectories within each MultiWaypointEnv
            for parent_dir in parent_dirs:
                subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
                multiwaypoint_dirs.extend(subdirs)
    
    if not multiwaypoint_dirs:
        print("✗ No MultiWaypointEnv timestamped directories found")
        return False
    
    # Get the most recent directory
    latest_dir = max(multiwaypoint_dirs, key=lambda d: d.stat().st_mtime)
    
    # Find the .mat file
    mat_files = list(latest_dir.glob("*.mat"))
    if not mat_files:
        print("✗ No .mat files found in the latest directory")
        return False
    
    mat_file = mat_files[0]
    print(f"✓ Found data file: {mat_file}")
    
    # Step 3: Run MATLAB analysis (if available)
    print("\n3. Running MATLAB analysis...")
    try:
        # Check if MATLAB is available
        result = subprocess.run(["matlab", "-help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("⚠ MATLAB not found in PATH, skipping MATLAB analysis")
            print("  You can run the analysis manually in MATLAB:")
            print(f"  >> analyze_multiwaypoint_mission('{mat_file.absolute()}')")
        else:
            # Run MATLAB analysis
            matlab_cmd = f"analyze_multiwaypoint_mission('{mat_file.absolute()}'); exit;"
            result = subprocess.run(["matlab", "-batch", matlab_cmd], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print("✓ MATLAB analysis completed successfully")
            else:
                print(f"⚠ MATLAB analysis had issues: {result.stderr}")
                print("  You can run the analysis manually in MATLAB:")
                print(f"  >> analyze_multiwaypoint_mission('{mat_file.absolute()}')")
    except subprocess.TimeoutExpired:
        print("⚠ MATLAB analysis timed out")
    except Exception as e:
        print(f"⚠ Error running MATLAB analysis: {e}")
    
    # Step 4: Summary
    print("\n4. Analysis complete!")
    print(f"   Data directory: {latest_dir}")
    print("   Generated files:")
    for file in latest_dir.iterdir():
        if file.is_file():
            print(f"     - {file.name}")
    
    print("\n" + "="*60)
    print("ANALYSIS PIPELINE COMPLETED")
    print("="*60)
    
    return True

def setup_environment():
    """Check and setup the required environment"""
    
    print("Checking environment requirements...")
    
    # Check Python packages
    required_packages = ['numpy', 'matplotlib', 'scipy', 'torch', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing Python packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check if required files exist
    required_files = [
        'render_multiwaypoint_advanced.py',
        'analyze_multiwaypoint_mission.m'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Required file not found: {file}")
            return False
    
    print("✓ Environment check passed")
    return True

if __name__ == "__main__":
    print("Advanced Multiwaypoint Mission Analysis")
    print("=" * 40)
    
    if not setup_environment():
        sys.exit(1)
    
    if run_multiwaypoint_analysis():
        print("\n🎉 Analysis pipeline completed successfully!")
    else:
        print("\n❌ Analysis pipeline failed!")
        sys.exit(1)
