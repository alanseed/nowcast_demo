import logging 
import datetime 
import json
import copy 
from pathlib import Path
import xarray as xr 

from utils.nc_utils import read_qpe_netcdf
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils import DBTransformer

def file_exists(file_path: Path) -> bool:
    """Check if the given file path exists."""
    return file_path.is_file()

def load_config_file(json_path: str, mname:str, dname:str ):

    # Load the JSON content
    try:
        with open(json_path, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return
    
    # === Load products ===
    product_docs = config_data.get("products", [])
    if not isinstance(product_docs, list):
        raise ValueError("'products' must be a list")

    config = {} 
    domain = {}
    for doc in product_docs:
        product_name = doc.get("product") 
        if product_name == mname:
            config = copy.deepcopy(doc)  
            break 
    
    # === Load domains ===
    domain_docs = config_data.get("domains", [])
    if not isinstance(domain_docs, list):
        raise ValueError("'domains' must be a list")
    
    for doc in domain_docs:
        domain_name = doc.get("domain") 
        if domain_name == dname:
            domain = copy.deepcopy(doc)  
            break 

    return config, domain 
            

def main():
    
    # Make the configuration dictionaries 
    mname = "nowcast" 
    dname = "akl_rad"
    config_file = "nowcast_config.json" 
    try:
        config, domain = load_config_file(config_file, mname, dname) # type: ignore
    except ValueError as e:
           print(f"Invalid configuration {e}")
           exit()

    # Read in the netCDF file as xr dataset 
    filename = "data/akl_rad_qpe_20230127.nc"
    data = read_qpe_netcdf(filename) 
    if data is None:
        print(f"Error reading {filename}")
        exit()

    # Set up the base_time for the nowcast 
    base_time = datetime.datetime(year=2023,month=1,day=27,hour=5,minute=0,tzinfo=datetime.UTC) 
    timestep = config.get("timestep", 600) 
    prev_time = base_time - datetime.timedelta(seconds=timestep) 
    print(f"Generating nowcasts for base time {base_time}") 

    # Read in the two rain fields 
    rain_ds = data.sel(time=[prev_time,base_time]) 
    print(rain_ds.info())

    threshold = config.get("precip_threshold",1) 
    transformer = DBTransformer(threshold)  


if __name__ == "__main__":
    main()
