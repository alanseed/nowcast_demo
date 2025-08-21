# Contains: get_db, get_config, get_parameters_df, get_parameters, to_utc_naive
from typing import Optional, Tuple
import datetime
import logging
import gridfs
from pymongo.database import Database
from pymongo import ASCENDING
from rainfields_db import read_netcdf_buffer

def ensure_utc(dt):
    if dt and dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt

def is_valid_iso8601(time_str: str) -> bool:
    try:
        datetime.datetime.fromisoformat(time_str)
        return True
    except ValueError:
        return False

def get_config(db:Database, product:str ,name:str) ->Tuple:
    """Return the model and domain configurations 

    Args:
        db (Database): Mongo Database
        product (str): Name of the product or model
        name (str): Name of the domain 

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        Tuple: (config, domain)
    """

    config_coll = db["config"]
    query = {"_id": f"{product}.{name}"}
    config = config_coll.find_one(query)
    if config is None:
        raise RuntimeError(f"Configuration not found for product {product}")

    domain_coll = db["domains"]
    domain = domain_coll.find_one({"name": f"{name}"})
    if domain is None:
        raise RuntimeError(f"Configuration not found for domain {name}")
    return config, domain

def get_rain_field(db, file_id):
    """Retrieve a specific rain field NetCDF file from GridFS and return as numpy array"""
    fs = gridfs.GridFS(db, collection='rain')
    file_obj = fs.get(file_id)
    data_bytes = file_obj.read()
    rain_rate, valid_time = read_netcdf_buffer(data_bytes)
    return rain_rate, valid_time

def get_rain_ids(query: dict, db: Database, name: str):
    coll_name = "rain.files"
    meta_coll = db[coll_name]

    # Fetch matching filenames and metadata in a single query
    fields_projection = {"_id": 1, "filename": 1, "metadata": 1}
    results = meta_coll.find(query, projection=fields_projection).sort(
        "filename", ASCENDING)
    files = []
    for doc in results:
        record = {"_id": doc["_id"],
                  "valid_time": doc["metadata"]["valid_time"]}
        files.append(record)
    return files

def get_base_time(valid_time:datetime.datetime, product:str, name:str, db:Database) -> Optional[datetime.datetime]:
    # Get the base_time for the nwp run nearest to the valid_time in UTC zone
    # Assume spin-up of 6 hours 
    start_base_time = valid_time - datetime.timedelta(hours=30)
    end_base_time = valid_time - datetime.timedelta(hours=6)
    base_time_query = {
        "metadata.domain":name,
        "metadata.product": product,
        "metadata.base_time": {"$gte": start_base_time, "$lte": end_base_time}
    }
    col_name = "rain.files"
    nwp_base_times = db[col_name].distinct(
        "metadata.base_time", base_time_query)

    if not nwp_base_times:
        logging.warning(
            f"Failed to find {product} data for {valid_time}")
        return None 

    nwp_base_times.sort(reverse=True)
    base_time = nwp_base_times[0]

    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=datetime.timezone.utc)

    return base_time