# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:25:35 2021

@author: Beau.Uriona and Nick Steele
"""

import sqlite3
from sqlite3 import Error as SQLError
from os import path, getenv
import pandas as pd
import requests

DB_DIR = path.dirname(path.realpath(__file__))
API_SERVER = getenv("API_SERVER", "https://wcc.sc.egov.usda.gov/awdbRestApi")
NETWORKS = ("SNTL", "SNTLT")


def get_meta(networks=NETWORKS, api_domain=API_SERVER):
    """
    Fetch station metadata from AWDB REST API v1
    New API endpoint: /services/v1/stations
    """
    dfs = []
    
    for network in networks:
        print(f"  Getting meta for {network} sites...")
        endpoint = "/services/v1/stations"
        
        # Build query parameters for the new API
        params = {
            "stationTriplets": f"*:*:{network}",  # wildcard for all stations in network
            "activeOnly": "false",  # Include inactive stations
            "returnStationElements": "false"  # We don't need element details
        }
        
        url = f"{api_domain}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            if response.ok:
                stations = response.json()
                
                if stations:
                    df = pd.DataFrame(stations)
                    dfs.append(df)
                else:
                    print(f"    No data returned for {network}")
            else:
                print(f"    Error fetching {network}: {response.status_code}")
        except Exception as e:
            print(f"    Exception fetching {network}: {e}")
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        return pd.DataFrame()


def parse_label(row):
    return f"{row['name']} ({row['station_id']}) ({row['elevation']}')"


def parse_triplet(triplet, index):
    trip_arr = triplet.split(":")
    return trip_arr[index]


def format_meta(df):
    """
    Format metadata from the new API structure to match the old database schema
    New API field mapping:
    - stationTriplet (unchanged)
    - name (unchanged)
    - elevation (unchanged)
    - latitude (unchanged)
    - longitude (unchanged)
    - huc (unchanged)
    - beginDate (unchanged)
    """
    if df.empty:
        return df
        
    df["station_id"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 0))
    df["state"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 1))
    df["network"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 2))
    df["label"] = df.apply(parse_label, axis=1)
    df.rename(columns={"stationTriplet": "triplet"}, inplace=True)
    
    dtype_dict = {
        "name": str,
        "label": str,
        "station_id": int,
        "state": str,
        "network": str,
        "triplet": str,
        "elevation": float,  # Changed from int to float to handle decimal elevations
        "latitude": float,
        "longitude": float,
        "huc": str,
        "beginDate": str,
    }
    
    # Select only the columns we need
    available_cols = [col for col in dtype_dict.keys() if col in df.columns]
    df = df[available_cols].astype(dtype_dict)
    return df


def convert_to_sqlite(df, db_path=path.join(DB_DIR, "meta.db")):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(name="meta", con=conn, if_exists="replace")
    except SQLError as e:
        print(f"Error converting metadata to sqlite - {e}")
    finally:
        if conn:
            conn.close()


def refresh_meta():
    print("Refreshing metadata/site list database...")
    raw_df = get_meta()
    
    if raw_df.empty:
        print("  ERROR: No metadata retrieved!")
        return
        
    df = format_meta(raw_df).sort_values(by="name", axis=0)
    csv_path = path.join(DB_DIR, "meta.csv")
    df.to_csv(csv_path, index=False)
    db_path = path.join(DB_DIR, "meta.db")
    convert_to_sqlite(df, db_path)
    print("  \nSuccess!")
    print(f"  Retrieved {len(df)} stations")


if __name__ == "__main__":
    refresh_meta()
