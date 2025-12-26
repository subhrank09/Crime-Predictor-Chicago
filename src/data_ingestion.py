import pandas as pd
import os

from sodapy import Socrata

# Create the directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# 1. Initialize the client
# Replace 'YOUR_APP_TOKEN' with the token you generated
client = Socrata("data.cityofchicago.org", "BnUcfauRP9YvuWS6Wtffw3vMm")

# 2. Define your query (SoQL - Socrata Query Language)
# Since the dataset is 8M+ rows, we filter for recent data to stay efficient
query = """
SELECT 
    id, date, primary_type, description, location_description, 
    arrest, domestic, district, ward, latitude, longitude
WHERE 
    date > '2024-01-01T00:00:00'
ORDER BY 
    date DESC
LIMIT 50000
"""

# 3. Fetch data
print("Connecting to Chicago API...")
results = client.get("ijzp-q8t2", query=query)

# 4. Convert to DataFrame
crime_df = pd.DataFrame.from_records(results)

# 5. Save to your directory
crime_df.to_json("data/raw/api_recent.json", orient="records", indent=4)
print(f"Downloaded {len(crime_df)} records to data/raw/api_recent.json")