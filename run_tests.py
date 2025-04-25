#!/usr/bin/env python3
"""
Simple script to run the SaF memory tests.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run SaF Memory tests')
    parser.add_argument('test', choices=['setup', 'dependencies', 'clustering', 'all'], 
                      help='Which test to run: "setup", "dependencies", "clustering", or "all" to run all tests')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    tests_dir = script_dir / 'tests'
    
    # Define the test scripts
    test_scripts = {
        'setup': tests_dir / 'test_setup.py',
        'dependencies': tests_dir / 'dependency_test.py',
        'clustering': tests_dir / 'test_clustering.py'
    }
    
    if args.test == 'all':
        print("Running all tests...")
        run_all = True
        tests_to_run = test_scripts.values()
    else:
        run_all = False
        tests_to_run = [test_scripts[args.test]]
        print(f"Running the {args.test} test...")
    
    success = True
    for script_path in tests_to_run:
        try:
            print(f"\nExecuting {script_path.name}...")
            subprocess.run([sys.executable, script_path], check=True)
            if not run_all:
                print(f"\n✅ {args.test.capitalize()} test completed successfully!")
        except subprocess.CalledProcessError:
            print(f"\n❌ {script_path.name} failed. Please check the error messages above.")
            success = False
        except FileNotFoundError:
            print(f"\n❌ Error: Could not find the test script at {script_path}")
            success = False
    
    if run_all and success:
        print("\n✅ All tests completed successfully!")
    elif run_all:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please specify which test to run:")
        print("  python run_tests.py setup       - Test the memory system setup")
        print("  python run_tests.py dependencies - Run the dependency test")
        print("  python run_tests.py clustering  - Run the clustering test")
        print("  python run_tests.py all         - Run all tests")
        sys.exit(1)
    
    main() 