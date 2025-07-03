# File: flows/b_training_gnb.py
import io
import pandas as pd
from domino_data.data_sources import DataSourceClient


def load_from_data_source(filename, datasource_name="credit_card_fraud_detection"):
    """Load DataFrame from Domino Data Source."""
    try:
        # Get the data source client
        ds_client = DataSourceClient()
        ds = ds_client.get_datasource(datasource_name)
        
        # Download the file
        print(f"📥 Downloading {filename} from data source")
        buf = io.BytesIO()
        ds.download_fileobj(filename, buf)
        buf.seek(0)
        
        # Load as DataFrame
        df = pd.read_csv(buf)
        print(f"✅ Loaded DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading from data source: {e}")
        raise

