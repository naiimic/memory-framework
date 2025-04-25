#!/usr/bin/env python3
"""
Simple script to run the SaF memory examples.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run SaF Memory examples')
    parser.add_argument('example', choices=['story', 'interactive'], 
                      help='Which example to run: "story" for story_memory.py, "interactive" for interactive_memory.py')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    examples_dir = script_dir / 'examples'
    
    if args.example == 'story':
        script_path = examples_dir / 'story_memory.py'
        print("Running the Story Memory example...")
        run_examples([script_path], "Story Memory")
    elif args.example == 'interactive':
        script_path = examples_dir / 'interactive_memory.py'
        print("Running the Interactive Memory Demo...")
        run_examples([script_path], "Interactive Memory")

def run_examples(script_paths, example_name):
    """Run the specified example scripts."""
    success = True
    for script_path in script_paths:
        print(f"\nExecuting {script_path.name}...")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Example execution failed for {script_path.name}. Please check the error messages above.")
            success = False
        except FileNotFoundError:
            print(f"❌ Error: Could not find the example script at {script_path}")
            success = False
    
    if success and len(script_paths) > 1:
        print(f"\n✅ All examples completed successfully!")
    elif success:
        print(f"\n✅ {example_name} example completed successfully!")
    else:
        print(f"\n❌ Some examples failed. Please check the error messages above.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please specify which example to run:")
        print("  python run_examples.py story       - Run the Story Memory example")
        print("  python run_examples.py interactive - Run the Interactive Memory Demo")
        sys.exit(1)
    
    main() 