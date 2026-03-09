#!/usr/bin/env python3
"""
Pipeline script to collect firm data and build database
Usage: python run_pipeline.py [--ceidg-api-key KEY] [--bbox south west north east]
"""

import argparse
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and check for errors"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run firm data collection pipeline')
    parser.add_argument('--ceidg-api-key', help='CEIDG API key (optional)')
    parser.add_argument('--bbox', nargs=4, type=float,
                       default=[50.02, 20.00, 52.05, 21.10],
                       help='Bounding box: south west north east (default: Krakow city center area)')
    parser.add_argument('--skip-osm', action='store_true', help='Skip OSM data collection')
    parser.add_argument('--skip-ceidg', action='store_true', help='Skip CEIDG data collection')

    args = parser.parse_args()

    # Check if we're in the right directory
    if not os.path.exists('collect_firm_data.py'):
        print("Error: Please run this script from the 'firmy' directory")
        sys.exit(1)

    success = True

    # Step 1: Collect OSM data
    if not args.skip_osm:
        bbox_str = f"{args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]}"
        cmd = f'python collect_firm_data.py --bbox {bbox_str}'
        success &= run_command(cmd, "Collecting OSM data")

    # Step 2: Collect CEIDG data
    if not args.skip_ceidg:
        cmd = 'python collect_ceidg_data.py'
        if args.ceidg_api_key:
            cmd += f' --api-key {args.ceidg_api_key}'
        success &= run_command(cmd, "Collecting CEIDG data")

    # Step 3: Build unified database
    success &= run_command('python build_database.py', "Building unified database")

    if success:
        print("\n🎉 Pipeline completed successfully!")
        print("Generated files:")
        print("- firmy.db (SQLite database)")
        print("- firmy_unified.csv (unified dataset)")
        print("- firmy_optimization.csv (optimization dataset)")
        print("- firmy_report.json (summary report)")
        print("\nYou can now run the optimization notebook: firmy.ipynb")
    else:
        print("\n❌ Pipeline failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()