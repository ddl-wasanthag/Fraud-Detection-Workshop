from pathlib import Path
from pathlib import Path
import os

def explore_directory(path, max_depth=3, current_depth=0):
    """Recursively explore directory structure"""
    try:
        path = Path(path)
        if not path.exists():
            print(f"❌ {path} doesn't exist")
            return
        
        if not path.is_dir():
            print(f"📄 {path} is a file, not directory")
            return
            
        print(f"📁 {path} ({'empty' if not any(path.iterdir()) else 'has contents'})")
        
        if current_depth >= max_depth:
            return
            
        for item in sorted(path.iterdir()):
            indent = "  " * (current_depth + 1)
            if item.is_dir():
                print(f"{indent}📁 {item.name}/")
                explore_directory(item, max_depth, current_depth + 1)
            else:
                size = item.stat().st_size
                print(f"{indent}📄 {item.name} ({size:,} bytes)")
                
    except PermissionError:
        print(f"❌ Permission denied: {path}")
    except Exception as e:
        print(f"❌ Error exploring {path}: {e}")

def find_similar_files(base_path, target_name="clean_cc_transactions.csv"):
    """Find files with similar names"""
    try:
        base = Path(base_path)
        if not base.exists():
            return []
            
        matches = []
        for file in base.rglob("*"):
            if file.is_file():
                name = file.name.lower()
                target = target_name.lower()
                
                # Check for partial matches
                if any(word in name for word in ["fraud", "transaction", "cc", "credit"]):
                    matches.append(file)
                elif target in name or name in target:
                    matches.append(file)
                    
        return matches
    except Exception as e:
        print(f"Error searching: {e}")
        return []

# Main exploration
print("=== Exploring /mnt/data ===")
explore_directory("/mnt/data")

print("\n=== Looking for similar files ===")
similar = find_similar_files("/mnt/data")
if similar:
    for file in similar:
        print(f"🔍 Found: {file}")
else:
    print("No similar files found")

# Also check current working directory
print(f"\n=== Current directory: {os.getcwd()} ===")
explore_directory(".", max_depth=2)


a = 33
b = 21

# Calculate sum
sum = int(a) + int(b)
print(f"The sum of {a} + {b} is {sum}")

# Write output
Path("/workflow/outputs/sum").write_text(str(sum))