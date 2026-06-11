import pandas as pd
from pathlib import Path
import sys

def standardize_to_interim(filename: str):
    """
    Converts raw XLSX or CSV files from the root 'data/raw/' folder into a standard 
    format, then drops it into the root 'data/interim/' folder.
    """
    # Force path resolution relative to this project's structure
    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent  # Moves up from 'src/' to the project root
    
    raw_path = root_path / "data" / "raw" / filename
    interim_dir = root_path / "data" / "interim"
    
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print(f"❌ Error: Could not find raw file at {raw_path}")
        sys.exit(1)
        
    print(f"🔄 Ingesting raw file: {raw_path.name}...")
    
    # 1. Read files according to their extension natively
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(raw_path, engine='openpyxl')
    else:
        df = pd.read_csv(raw_path)
        
    # 2. Match the exact column names your downstream pipeline expects
    mapping = {
        'Sample time': 'Sample time',
        'AE_POWER(kW)': 'AE_POWER(kW)',
        'HEADING(degree)': 'HEADING(degree)',
        'SHIP SPEED(knots)': 'SPEED(knots)'  # Harmonizes 'SHIP SPEED(knots)' to standard 'SPEED(knots)'
    }
    df = df.rename(columns=mapping)
    
    # Dynamic detection of whichever column acts as the timestamp index
    time_col = next((c for c in df.columns if 'SAMPLE TIME' in c.upper() or 'TIME' in c.upper()), None)
    
    if time_col is None:
        print(f"❌ Error: Could not find a timestamp column. Available columns: {list(df.columns)}")
        sys.exit(1)
        
    # Rename it cleanly to 'Sample time' for consistency across downstream blocks
    df = df.rename(columns={time_col: 'Sample time'})
    
    # 3. Clean up footnotes and metadata text block rows from the bottom
    df = df.dropna(subset=['Sample time'])
    df['Sample time'] = df['Sample time'].astype(str).str.strip()
    
    # Drop rows matching spreadsheet footnote definitions (e.g., 'Tag Name', 'Definition')
    df = df[~df['Sample time'].str.contains('Tag Name|Sample time|Definition|^,+$', case=False, na=False)]
    
    # 4. Convert strings to valid datetimes & sort chronologically
    try:
        df['Sample time'] = pd.to_datetime(df['Sample time'], errors='coerce')
        df = df.dropna(subset=['Sample time']) # Trims away any row that failed date conversion
        df = df.sort_values('Sample time').set_index('Sample time')
    except Exception as e:
        print(f"❌ Date parsing failure: {e}")
        sys.exit(1)
        
    # 5. Save uniform file to Interim Storage
    # Changing the raw file extension (.xlsx or .csv) to clean interim .csv format
    out_name = Path(filename).stem + ".csv"
    output_path = interim_dir / out_name
    df.to_csv(output_path)
    
    print(f"✅ Preprocessing complete! Cleaned file saved to: data/interim/{out_name}\n")
    return out_name

# --- EXECUTABLE INTERFACE ---
if __name__ == "__main__":
    # Fallback default file if no arguments are specified in terminal
    default_file = "Wembley_voy_236.csv" 
    
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    standardize_to_interim(target_file)