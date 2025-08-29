import logging
import datetime
import json
import copy
import numpy as np
from pathlib import Path

from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.motion import get_method as get_oflow_method
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils.transformer import DBTransformer 

from pysteps.param.nc_utils import read_qpe_netcdf
from pysteps.param.steps_params import StepsParameters
from pysteps.param.shared_utils import calculate_parameters

def file_exists(file_path: Path) -> bool:
    """Check if the given file path exists."""
    return file_path.is_file()


def load_config_file(json_path: str, mname: str, dname: str):

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
        domain_name = doc.get("name")
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
        config, domain = load_config_file(
            config_file, mname, dname)  # type: ignore
    except ValueError as e:
        print(f"Invalid configuration {e}")
        exit()

    # Read in the netCDF file as xr dataset 
    file_path = "/home/alanseed/alan/nowcast_demo/data/akl_rad_qpe_20230127.nc"
    data = read_qpe_netcdf(file_path)
    if data is None:
        print(f"Error reading {file_path}")
        exit()

    # Set up the base_time for the nowcast
    base_time = datetime.datetime(
        year=2023, month=1, day=27, hour=5, minute=0, tzinfo=datetime.UTC)
    seconds = int(config.get("timestep", 600))
    minutes = seconds // 60
    dt = datetime.timedelta(seconds=seconds)

    # AR(2) model
    times = [base_time+ia*dt for ia in range(-2, 1)]
    print(f"Generating nowcasts for base time {base_time}")

    threshold = config.get("precip_threshold", 1)
    transformer = DBTransformer(threshold)
    zero_value = transformer.zerovalue

    db_fields = []
    for time in times:
        field = data.rain.sel(time=time).values
        db_field = transformer.transform(field)
        db_fields.append(db_field)
    db_fields = np.stack(db_fields)

    n_rows = domain.get("n_rows")
    n_cols = domain.get("n_cols")
    psize = domain.get("p_size", 1000)
    n_levels = config.get("n_cascade_levels")
    scale_break = config.get("scale_break") / psize

    # Calculate the optical flow
    of_method = get_oflow_method("lucaskanade")
    oflow = of_method(db_fields)

    # Decompose the t-2,t-1,t dbr fields
    bp_filter = filter_gaussian((n_rows, n_cols), n_levels)
    cascades = []
    for ia in range(3):
        cascade = decomposition_fft(db_fields[ia, :, :], bp_filter,
                                    normalize=True, compute_stats=True)
        cascades.append(cascade)

    # calculate the parameters
    params = calculate_parameters(
        db_fields[2, :, :], cascades, oflow, scale_break, zero_value, minutes)

    params.set_metadata("valid_time", base_time) 
    params.set_metadata("domain", dname)
    params.set_metadata("product", "qpe") 



if __name__ == "__main__":
    main()
