import pandas as pd
from pathlib import Path
import sys

def standardize_to_interim(filename: str):
    script_dir = Path(__file__).resolve().parent
    root_path = script_dir.parent
    
    raw_path = root_path / "data" / "raw" / filename
    interim_dir = root_path / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print(f"❌ Error: Could not find raw file at {raw_path}")
        sys.exit(1)
        
    print(f"🔄 Ingesting raw file: {raw_path.name}...")
    
    if filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(raw_path, engine='openpyxl')
    else:
        df = pd.read_csv(raw_path)
        
    mapping = {
        'Sample time': 'Sample time',
        'AE_POWER(kW)': 'AE_POWER(kW)',
        'HEADING(degree)': 'HEADING(degree)',
        'SHIP SPEED(knots)': 'SPEED(knots)' 
    }
    df = df.rename(columns=mapping)
    
    time_col = next((c for c in df.columns if 'SAMPLE TIME' in c.upper() or 'TIME' in c.upper()), None)
    if time_col is None:
        print("❌ Error: Could not find a timestamp column.")
        sys.exit(1)
    df = df.rename(columns={time_col: 'Sample time'})
    
    df = df.dropna(subset=['Sample time'])
    df['Sample time'] = df['Sample time'].astype(str).str.strip()
    df = df[~df['Sample time'].str.contains('Tag Name|Sample time|Definition|^,+$', case=False, na=False)]
    
    try:
        import re
        first_date_val = str(df['Sample time'].iloc[0]).strip()
        
        # Check if the date starts with a 4-digit year (ISO Format like 2026-01-01)
        if re.match(r'^\d{4}', first_date_val):
            # Parse natively without forcing dayfirst
            df['Sample time'] = pd.to_datetime(df['Sample time'], errors='coerce')
        else:
            # Parse as European format (DD/MM/YYYY). 
            # format='mixed' suppresses the UserWarning and optimizes the fallback parser
            df['Sample time'] = pd.to_datetime(
                df['Sample time'], 
                dayfirst=True, 
                format='mixed', 
                errors='coerce'
            )
            
        df = df.dropna(subset=['Sample time']) # Trims away any row that failed date conversion
        df = df.sort_values('Sample time').set_index('Sample time')
    except Exception as e:
        print(f"❌ Date parsing failure: {e}")
        sys.exit(1)

    # FIX: Handle BOTH string letters and their ASCII integer codes
    if 'LAT-DEG(degree)' in df.columns and 'LAT-MIN(min)' in df.columns:
        lat = df['LAT-DEG(degree)'].astype(float) + df['LAT-MIN(min)'].astype(float) / 60.0
        if 'LAT-NS' in df.columns:
            # 83 is ASCII for 'S'
            lat = lat * df['LAT-NS'].apply(lambda x: -1 if str(x).strip().upper() == 'S' or str(x).strip() == '83' else 1)
        df['LATITUDE(DD)'] = lat

    if 'LONG-DEG(degree)' in df.columns and 'LONG-MIN(min)' in df.columns:
        lon = df['LONG-DEG(degree)'].astype(float) + df['LONG-MIN(min)'].astype(float) / 60.0
        if 'LONG-EW' in df.columns:
            # 87 is ASCII for 'W'
            lon = lon * df['LONG-EW'].apply(lambda x: -1 if str(x).strip().upper() == 'W' or str(x).strip() == '87' else 1)
        df['LONGITUDE(DD)'] = lon
        
    out_name = Path(filename).stem + ".csv"
    output_path = interim_dir / out_name
    df.to_csv(output_path)
    
    print(f"✅ Preprocessing complete! Cleaned file saved to: data/interim/{out_name}\n")
    return out_name

if __name__ == "__main__":
    default_file = "File completo.xlsx" 
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    standardize_to_interim(target_file)