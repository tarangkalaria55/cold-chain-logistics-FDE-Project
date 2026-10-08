import sys
from pathlib import Path
import pandas as pd

# __file__ is 'experiment/phase-0/ingest_legacy_data.py'
script_dir = Path(__file__).resolve().parent # points to experiment/phase-0
project_root = script_dir.parents[0] # climbs up 1 levels to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import reveal, settings
from src.db import make_engine

data_path = project_root / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"

db_user = settings.sql_admin_user
db_password = reveal(settings.sql_admin_password)

# 1. Load the raw dataset
print(f"Loading CSV from {data_path}...")
df = pd.read_csv(data_path)

# 2. Map clean columns to a messy 2000s legacy enterprise schema
legacy_mapping = {
    'timestamp': 'TS_UTC',
    'vehicle_gps_latitude': 'V_LAT',
    'vehicle_gps_longitude': 'V_LON',
    'iot_temperature': 'IOT_TEMP_VAL_C',
    'cargo_condition_status': 'CGO_COND_CD',
    'risk_classification': 'RISK_CLS_TXT',
    'delay_probability': 'DELAY_PROB_DEC',
    'port_congestion_level': 'PRT_CNG_LVL',
    'route_risk_level': 'RT_RSK_IDX'
}

# Keep only the columns we mapped for this demo and rename them
df_legacy = df[list(legacy_mapping.keys())].rename(columns=legacy_mapping)

# Add a fake ingestion flag to make it look like an automated legacy system
df_legacy['SYS_INGEST_FLAG'] = 'Y'

# 3. Connect to Docker MSSQL Server
print("Connecting to legacy MSSQL Database...")
# Requires ODBC Driver 18 for SQL Server on the host OS.
# fast_executemany sends rows in bulk (far faster); no query timeout for a long load.
engine = make_engine(db_user, db_password, fast_executemany=True, query_timeout=None)

# 4. Ingest data into the messy table name
table_name = 'TBL_SC_FLEET_HIST_RAW'
print(f"Ingesting into {table_name}. This may take a minute...")
# One transaction: if the load fails midway the old table is kept instead of being left dropped/empty.
try:
    with engine.begin() as conn:
        df_legacy.to_sql(table_name, conn, if_exists='replace', index=False, schema='dbo', chunksize=1000)
finally:
    engine.dispose()

print("✅ Legacy data ingestion complete!")