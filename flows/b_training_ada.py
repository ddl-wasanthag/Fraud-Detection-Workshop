import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from flows.generic_trainer import train_fraud
from helpers.domino_short_id import domino_short_id


def debug_file_system():
    """Debug function to understand the file system state."""
    print("=== FILE SYSTEM DEBUG ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Home directory: {os.path.expanduser('~')}")
    
    # Check various mount points
    mount_points = ["/mnt", "/mnt/data", "/mnt/artifacts", "/workflow", "/tmp"]
    for mount in mount_points:
        if os.path.exists(mount):
            print(f"\n📂 {mount}:")
            try:
                items = os.listdir(mount)
                for item in items[:10]:  # Show first 10 items
                    item_path = os.path.join(mount, item)
                    if os.path.isdir(item_path):
                        print(f"  📁 {item}/")
                    else:
                        size = os.path.getsize(item_path)
                        print(f"  📄 {item} ({size:,} bytes)")
                if len(items) > 10:
                    print(f"  ... and {len(items) - 10} more items")
            except PermissionError:
                print("  ❌ Permission denied")
        else:
            print(f"❌ {mount} not found")
    
    # Look for CSV files specifically
    print("\n=== SEARCHING FOR CSV FILES ===")
    search_paths = ["/mnt/data", "/tmp", "/workflow"]
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"\n🔍 Searching in {search_path}:")
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.csv'):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        print(f"  📄 {full_path} ({size:,} bytes)")


def train_fraud_ada(clean_filepath, random_state=None):
    """Train AdaBoost classifier for fraud detection."""
    model_obj = AdaBoostClassifier(
        n_estimators=10,
        learning_rate=0.1,
        algorithm="SAMME",
        random_state=random_state
    )
    
    model_name = "AdaBoost"
    experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
    
    return train_fraud(model_obj, model_name, clean_filepath, experiment_name)


def main():
    # Debug the file system first
    debug_file_system()
    
    # Read the preprocessed data path from workflow input
    input_file = "/workflow/inputs/preprocessed_df_path"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Available files in /workflow/inputs:")
        if os.path.exists("/workflow/inputs"):
            for file in os.listdir("/workflow/inputs"):
                print(f"  - {file}")
        return "Error: Input file not found"
    
    preprocessed_df_path = Path(input_file).read_text().strip()
    print(f'\n📍 Processing data from: {preprocessed_df_path}')
    
    # Check if the file exists
    if not os.path.exists(preprocessed_df_path):
        print(f"❌ Data file not found: {preprocessed_df_path}")
        
        # Try to find it in other locations
        filename = os.path.basename(preprocessed_df_path)
        print(f"🔍 Looking for {filename} in other locations...")
        
        search_locations = [
            "/mnt/data",
            "/tmp",
            "/workflow/inputs",
            "/workflow/outputs"
        ]
        
        found_file = None
        for location in search_locations:
            potential_path = os.path.join(location, filename)
            if os.path.exists(potential_path):
                print(f"✅ Found file at: {potential_path}")
                found_file = potential_path
                break
        
        if found_file:
            preprocessed_df_path = found_file
        else:
            print("❌ File not found in any search location")
            return "Error: Data file not found"
    
    # Try to read the file
    try:
        df = pd.read_csv(preprocessed_df_path)
        print(f'✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns')
        print(f'Columns: {list(df.columns)}')
        print(f'First few rows:\n{df.head()}')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return f"Error reading file: {e}"
    
    # Train the model
    try:
        result = train_fraud_ada(preprocessed_df_path)
        print(f"✅ Training completed successfully")
        
        # Write output
        Path("/workflow/outputs/results_df").write_text(result)
        return result
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return f"Training failed: {e}"


if __name__ == "__main__":
    main()