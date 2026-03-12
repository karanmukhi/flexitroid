from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import requests
import datetime
from datetime import datetime, timedelta


def fetch_day_ahead_price_curves(start_date, end_date):
    url = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
    headers = {"Accept": "text/plain"}


    responses = []  # to store responses from each request
    current_start = start_date

    while current_start <= end_date:
        # Calculate block end (maximum of 7 days per request)
        current_end = current_start + timedelta(days=6)
        if current_end > end_date:
            current_end = end_date

        # Format dates as required: "YYYY-MM-DDT00:00Z"
        # (Assuming the API expects these strings exactly.)
        from_str = current_start.strftime("%Y-%m-%dT00:00Z")
        to_str = current_end.strftime("%Y-%m-%dT23:30Z")

        print(f"Requesting data from {from_str} to {to_str}")

        params = {"from": from_str, "to": to_str, "dataProviders": "APXMIDP"}

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            responses.append(response.json()["data"])
        else:
            print(
                f"Request from {from_str} to {to_str} failed with status code {response.status_code}"
            )

        # Move to the next block: day after current_end
        current_start = current_end + timedelta(days=1)

    # Do something with the collected responses

    data = [data_point for block_response in responses for data_point in block_response]
    data = data[:]
    df = pd.DataFrame(data)
    df = df.sort_values(by="startTime")
    df = df[:-1]
    grouped = df.groupby("settlementDate")
    day_curves = {}
    for group_name, group_data in grouped:
        day_curves[group_name] = group_data["price"].values
    return day_curves



def clean_response(response_json):
    costs = []
    for day, curve in response_json.items():
        if len(curve) != 48:
            print(f"Skipping day {day} with {len(curve)} values")
            continue
        costs.append(curve)
    costs = np.array(costs)
    return costs


def save_resposne(costs, filename):
    np.savetxt(filename, costs, delimiter=",", fmt="%.6f")

def cache_day_ahead_price_curves(start_date, end_date, filename=None):
    response_json = fetch_day_ahead_price_curves(start_date, end_date)
    costs = clean_response(response_json)
    if filename is None:
        filename = f"day_ahead_price_curves_{start_date.strftime('%Y-%m-%dT00:00Z')}_to_{end_date.strftime('%Y-%m-%dT23:30Z')}.csv"
    save_resposne(costs, filename)
