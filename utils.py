#!./.venv/bin/python
# coding: utf-8

from datetime import timedelta
from os import path, getenv
from functools import reduce

import requests
import pandas as pd
from requests_cache import CachedSession

API_SERVER = getenv("API_SERVER", "https://wcc.sc.egov.usda.gov/awdbRestApi")
NULL_OPTION = {"label": "", "value": None}
THIS_DIR = path.dirname(path.realpath(__file__))
DB_DIR = path.join(THIS_DIR, "dbs")
CACHE_PATH = getenv("CACHE_PATH", path.join(DB_DIR, ".cache.db"))
CACHE_REFRESH = timedelta(hours=12)
CACHE_ARGS = {
    "cache_name": CACHE_PATH,
    "backend": "sqlite",
    "expire_after": CACHE_REFRESH,
}


def add_null_option(options=None):
    if options:
        options.insert(0, NULL_OPTION)
        return options
    return [NULL_OPTION]

   
def get_singlestation_data(stationtriplet, element, s_date, e_date, orient, sesh=None):

    def get_data(stationtriplet, element, s_date, e_date, orient, server=API_SERVER):
        """
        Fetch data from the AWDB REST API v1
        New API format: /services/v1/data
        Parameters: stationTriplets, elements, duration, beginDate, endDate
        """
        endpoint = "/services/v1/data"
        
        # Build query parameters
        params = {
            "stationTriplets": stationtriplet,
            "elements": element,
            "duration": "DAILY",
            "beginDate": s_date,
            "endDate": e_date
        }
        
        url = f"{server}{endpoint}"
        print(f"getting data for {url}?{requests.compat.urlencode(params)}")
        
        if sesh:
            req = sesh.get(url, params=params)
        else:
            req = requests.get(url, params=params)
            
        if req.ok:
            response_data = req.json()
            
            # The new API returns data in a different structure:
            # [{"stationTriplet": "...", "data": [{"stationElement": {...}, "values": [...]}]}]
            
            if response_data and len(response_data) > 0:
                station_data = response_data[0]
                
                if "data" in station_data and len(station_data["data"]) > 0:
                    element_data = station_data["data"][0]
                    values = element_data.get("values", [])
                    
                    # Extract dates and values
                    dates = [v["date"] for v in values if "date" in v]
                    vals = [v.get("value") for v in values]
                    
                    df = pd.DataFrame({
                        "Date": dates,
                        f"{stationtriplet}({element})": vals
                    })
                    df.set_index("Date", inplace=True)
                    return df
                    
        # Return empty dataframe if no data
        return pd.DataFrame()

    # Create derived products here:
    if element == "WTEQ - Accumulative":
        element = 'WTEQ'
        df = get_data(stationtriplet, element, s_date, e_date, orient)
        
        if df.empty:
            return df
            
        # Categorize measurements by water year.
        df.reset_index(inplace=True)  
        pd.to_datetime(df['Date'])
        df['water_year'] = pd.to_datetime(df['Date']).dt.year.where(pd.to_datetime(df['Date']).dt.month < 10, pd.to_datetime(df['Date']).dt.year + 1)
        df['water_year'] = list(map(lambda x: str(x), df['water_year']))
        df.set_index('Date', inplace=True)
        # Take the difference of WTEQ measurements and change all neg delta to 0.
        df[f'{stationtriplet}(WTEQ - Accumulative)'] = df[f'{stationtriplet}(WTEQ)'].diff().clip(lower=0)
        # Groupby water year and then take the cumulative sum of the WTEQ measurements.
        df = pd.DataFrame(df.groupby(['water_year'])[f'{stationtriplet}(WTEQ - Accumulative)'].cumsum())
    
    else:
        df = get_data(stationtriplet, element, s_date, e_date, orient)

    return df


def get_multiplestation_data(stationparameter_pairs, s_date, e_date, orient="records"):

    with CachedSession(**CACHE_ARGS) as sesh:
        try:
            data = reduce(
                lambda left, right: pd.merge(
                    left, right, left_index=True, right_index=True, how="outer"
                ),
                [
                    get_singlestation_data(
                        stationtriplet=i[0],
                        element=i[1],
                        s_date=s_date,
                        e_date=e_date,
                        orient=orient,
                        sesh=sesh,
                    )
                    for i in stationparameter_pairs
                ],
            )

            return data

        except KeyError as err:
            print(f"KeyError occurred. - {err}")


def get_plot_config(img_filename="nrcs_chart.png"):
    return {
        "modeBarButtonsToRemove": ["sendDataToCloud", "lasso2d", "select2d"],
        "showAxisDragHandles": True,
        "showAxisRangeEntryBoxes": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "filename": img_filename,
            "width": 1200,
            "height": 700,
        },
        # 'scrollZoom': False,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
            "drawtext",
        ],
    }


if __name__ == "__main__":
    print("I do nothing, just non dynamic components that take up space...")
