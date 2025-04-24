#!/usr/bin/env python3
"""
Simple script to run the SaF memory examples.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run SaF Memory examples')
    parser.add_argument('example', choices=['story', 'interactive', 'test', 'check'], 
                      help='Which example to run: "story" for story_memory.py, "interactive" for interactive_memory.py, "test" to check setup, or "check" to run the dependency test')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    if args.example == 'story':
        script_path = script_dir / 'examples' / 'story_memory.py'
        print("Running the Story Memory example...")
    elif args.example == 'interactive':
        script_path = script_dir / 'examples' / 'interactive_memory.py'
        print("Running the Interactive Memory Demo...")
    elif args.example == 'check':
        script_path = script_dir / 'dependency_test.py'
        print("Running the dependency check...")
    else:  # args.example == 'test'
        script_path = script_dir / 'test_setup.py'
        print("Testing the memory system setup...")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError:
        print("Example execution failed. Please check the error messages above.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find the example script at {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please specify which example to run:")
        print("  python run_examples.py story        - Run the Story Memory example")
        print("  python run_examples.py interactive  - Run the Interactive Memory Demo")
        print("  python run_examples.py test         - Test the memory system setup")
        print("  python run_examples.py check        - Run the dependency test")
        sys.exit(1)
    
    main() 