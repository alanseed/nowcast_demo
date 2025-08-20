
import pandas as pd 
import datetime 
import logging 
from pymongo.database import Database 
from pymongo import ASCENDING
from rainfields_db import get_param_docs 
from models.steps_params import StepsParameters 

def get_parameters_df(query: dict, param_coll) -> pd.DataFrame:
    """
    Retrieve and interpret parameters as StepsParameters instances.
    """
    raw_docs = get_param_docs(query, param_coll)
    records = []

    for doc in raw_docs:
        metadata = doc.get("metadata", {})
        valid_time = metadata.get("valid_time")
        base_time = metadata.get("base_time")
        ensemble = metadata.get("ensemble")

        if valid_time and valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)
        if base_time and base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=datetime.timezone.utc)

        try:
            
            param = StepsParameters.from_dict(doc)
            records.append({
                "valid_time": valid_time,
                "base_time": base_time,
                "ensemble": ensemble,
                "param": param
            })
        except Exception as e:
            print(f"Warning: could not parse parameter for {valid_time}: {e}")

    return pd.DataFrame(records, columns=["valid_time", "base_time", "ensemble", "param"])


def make_storms_df(db: Database, name: str, threshold: float,
                   start_time: datetime.datetime,
                   end_time: datetime.datetime) -> pd.DataFrame:
    """Make a dataframe with start and end times of rain events
    where the wetted area ratio for the radar qpe exceeds a threshold.

    Args:
        db (Database): MongoDB database connection
        name (str): Domain or site name prefix for the collection
        threshold (float): Minimum wetted area ratio to classify as a storm
        start_time (datetime.datetime): Start of the search window (UTC)
        end_time (datetime.datetime): End of the search window (UTC)

    Returns:
        pd.DataFrame: DataFrame with ['start', 'end'] columns as timestamp objects (UTC)
    """

    coll = db[f"{name}.rain.files"]

    # Build query for fields with war above threshold
    query = {
        "metadata.product": "QPE",
        "metadata.wetted_area_ratio": {"$gte": threshold},
        "metadata.valid_time": {"$gte": start_time, "$lte": end_time},
    }
    projection = {"_id": 0, "metadata.valid_time": 1,
                  "metadata.wetted_area_ratio": 1}

    count = coll.count_documents(query)
    logging.info(f"Found {count} fields between {start_time} - {end_time}")

    # Build time series
    times = []
    values = []
    for doc in coll.find(filter=query, projection=projection).sort("metadata.valid_time", ASCENDING):
        vtime = pd.to_datetime(doc["metadata"]["valid_time"], utc=True)
        war = float(doc["metadata"]["wetted_area_ratio"])
        times.append(vtime)
        values.append(war)

    war_ts = pd.Series(values, index=times).sort_index()

    max_gap = pd.Timedelta(minutes=30)
    min_duration = pd.Timedelta(hours=1)

    storms = []
    if not war_ts.empty:
        start = war_ts.index[0]
        end = start
        for vtime, war in war_ts.items():
            if vtime - end > max_gap:  # type: ignore
                if end - start > min_duration:  # type: ignore
                    storms.append({"start": start, "end": end})
                start = vtime
                end = vtime
            else:
                end = vtime

        # Check the last segment
        if end - start > min_duration:  # type: ignore
            storms.append({"start": start, "end": end})

    storms_df = pd.DataFrame(storms)
    return storms_df


