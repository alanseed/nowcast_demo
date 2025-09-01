import logging
import datetime
import json
import copy
import netCDF4
from typing import Optional

from pyproj import CRS
import numpy as np
from pathlib import Path

from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.motion import get_method as get_oflow_method
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.utils.transformer import DBTransformer

from pysteps.param.nc_utils import read_qpe_netcdf
from pysteps.param.steps_params import StepsParameters
from pysteps.param.shared_utils import calculate_parameters, update_field


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


def write_netcdf(file_path: Path, rain: np.ndarray, geo_data: dict,
                 valid_times: list[datetime.datetime], ensembles: list[int] | None) -> None:
    """
    Write a set of rainfall grids to a CF-compliant NetCDF file using i2 data and scale_factor.

    Args:
        file_path (Path): Full path to the output file.
        rain (np.ndarray): Rainfall array. Shape is [ensemble, time, y, x] if ensembles is provided,
                           otherwise [time, y, x], with units in mm/h as float.
        geo_data (dict): Geospatial metadata (must include 'x', 'y', and optionally 'projection').
        valid_times (list[datetime.datetime]): List of timezone-aware valid times.
        ensembles (list[int] | None): Optional list of ensemble member IDs.
    """
    # Convert datetime to seconds since epoch
    time_stamps = [vt.timestamp() for vt in valid_times]

    x = geo_data["x"]
    y = geo_data["y"]
    projection = geo_data.get("projection", "EPSG:4326")
    rain_fill_value = -1

    with netCDF4.Dataset(file_path, mode="w", format="NETCDF4") as ds:
        # Define dimensions
        ds.createDimension("y", len(y))
        ds.createDimension("x", len(x))
        ds.createDimension("time", len(valid_times))
        if ensembles is not None:
            ds.createDimension("ensemble", len(ensembles))

        # Define coordinate variables
        x_var = ds.createVariable("x", "f4", ("x",))
        x_var[:] = x
        x_var.standard_name = "projection_x_coordinate"
        x_var.units = "m"

        y_var = ds.createVariable("y", "f4", ("y",))
        y_var[:] = y
        y_var.standard_name = "projection_y_coordinate"
        y_var.units = "m"

        t_var = ds.createVariable("time", "f8", ("time",))
        t_var[:] = time_stamps
        t_var.standard_name = "time"
        t_var.units = "seconds since 1970-01-01T00:00:00Z"
        t_var.calendar = "standard"

        if ensembles is not None:
            e_var = ds.createVariable("ensemble", "i4", ("ensemble",))
            e_var[:] = ensembles
            e_var.standard_name = "ensemble"
            e_var.units = "1"

        # Define the rainfall variable with proper fill_value
        rain_dims = ("time", "y", "x") if ensembles is None else (
            "ensemble", "time", "y", "x")
        rain_var = ds.createVariable(
            "rainfall", "i2", rain_dims,
            zlib=True, complevel=5, fill_value=rain_fill_value
        )

        # Scale and store rainfall
        rain_scaled = np.where(
            np.isnan(rain),
            rain_fill_value,
            np.round(rain * 10).astype(np.int16)
        )
        rain_var[...] = rain_scaled

        # Metadata
        rain_var.scale_factor = 0.1
        rain_var.add_offset = 0.0
        rain_var.units = "mm/h"
        rain_var.long_name = "Rainfall rate"
        rain_var.grid_mapping = "projection"
        rain_var.coordinates = " ".join(rain_dims)

        # CRS
        crs = CRS.from_user_input(projection)
        cf_grid_mapping = crs.to_cf()
        spatial_ref = ds.createVariable("projection", "i4")
        for key, value in cf_grid_mapping.items():
            setattr(spatial_ref, key, value)

        # Global attributes
        ds.Conventions = "CF-1.10"
        ds.title = "STEPS_param ensemble nowcast"
        ds.institution = ""
        ds.references = ""
        ds.comment = ""

def make_nc_name(domain: str, prod: str, valid_time: datetime.datetime,
                 base_time: Optional[datetime.datetime] = None, ens: Optional[int] = None,
                 name_template: Optional[str] = None) -> str:
    """
    Generate a unique name for a single rain field using a formatting template.

    Default templates:
        Forecast products: "$D_$P_$V{%Y%m%dT%H%M%S}_$B{%Y%m%dT%H%M%S}_$E.nc"
        QPE products: "$D_$P_$V{%Y%m%dT%H%M%S}.nc"

    Where:
        $D = Domain name
        $P = Product name
        $V = Valid time (with strftime format)
        $B = Base time (with strftime format)
        $E = Ensemble number (zero-padded 2-digit)

    Returns:
        str: Unique NetCDF file name.
    """

    if not isinstance(valid_time, datetime.datetime):
        raise TypeError(f"valid_time must be datetime, got {type(valid_time)}")

    if base_time is not None and not isinstance(base_time, datetime.datetime):
        raise TypeError(f"base_time must be datetime or None, got {type(base_time)}")

    # Default template logic
    if name_template is None:
        name_template = "$D_$P_$V{%Y-%m-%dT%H:%M:%S}"
        if base_time is not None:
            name_template += "_$B{%Y-%m-%dT%H:%M:%S}"
        if ens is not None:
            name_template += "_$E"
        name_template += ".nc"

    result = name_template

    # Ensure timezone-aware times
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)
    if base_time is not None and base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=datetime.timezone.utc)

    # Replace flags
    while "$" in result:
        flag_posn = result.find("$")
        if flag_posn == -1:
            break
        f_type = result[flag_posn + 1]

        try:
            if f_type in ['V', 'B']:
                field_start = result.find("{", flag_posn + 1)
                field_end = result.find("}", flag_posn + 1)
                if field_start == -1 or field_end == -1:
                    raise ValueError(f"Missing braces for format of '${f_type}' in template.")

                fmt = result[field_start + 1:field_end]
                if f_type == 'V':
                    time_str = valid_time.strftime(fmt)
                elif f_type == 'B' and base_time is not None:
                    time_str = base_time.strftime(fmt)
                else:
                    time_str = ""

                result = result[:flag_posn] + time_str + result[field_end + 1:]

            elif f_type == 'D':
                result = result[:flag_posn] + domain + result[flag_posn + 2:]
            elif f_type == 'P':
                result = result[:flag_posn] + prod + result[flag_posn + 2:]
            elif f_type == 'E' and ens is not None:
                result = result[:flag_posn] + f"{ens:02d}" + result[flag_posn + 2:]
            else:
                raise ValueError(f"Unknown or unsupported flag '${f_type}' in template.")
        except Exception as e:
            raise ValueError(f"Error processing flag '${f_type}': {e}")

    return result

def generate_geo_dict(domain):
    ncols = domain.get("n_cols")
    nrows = domain.get("n_rows")
    psize = domain.get("p_size")
    start_x = domain.get("start_x")
    start_y = domain.get("start_y")
    x = [start_x + i * psize for i in range(ncols)]
    y = [start_y + i * psize for i in range(nrows)]

    out_geo = {} 
    out_geo['x'] = x 
    out_geo['y'] = y
    out_geo['xpixelsize'] = psize
    out_geo['ypixelsize'] = psize
    out_geo['x1'] = start_x
    out_geo['y1'] = start_y
    out_geo['x2'] = start_x + (ncols-1)*psize
    out_geo['y2'] = start_y + (nrows - 1)*psize
    out_geo['projection'] = domain["projection"]["epsg"]
    out_geo["cartesian_unit"] = 'm',
    out_geo["yorigin"] = 'lower',
    out_geo["unit"] = 'mm/h'
    out_geo["threshold"] = 0
    out_geo["transform"] = None

    return out_geo

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
    db_threshold = 10 * np.log10(threshold)
    transformer = DBTransformer(threshold)
    zero_value = transformer.zerovalue

    db_fields = []
    for time in times:
        field = data.rain.sel(time=time).values
        db_field = transformer.transform(field)
        db_fields.append(db_field)
    db_fields = np.stack(db_fields)

    n_rows = db_fields.shape[1]
    n_cols = db_fields.shape[2]
    psize = domain.get("p_size", 1000)
    n_forecasts = config.get("n_forecasts", 12)
    n_ensembles = config.get("n_ensembles", 10)
    n_levels = config.get("n_cascade_levels", 6)
    scale_break = config.get("scale_break", 20000) / psize   # pixel units 
    scale_break_km = config.get("scale_break", 20000) / 1000 # km units 

    # Calculate the optical flow
    of_method = get_oflow_method("lucaskanade")
    oflow = of_method(db_fields)

    # Decompose the t-2,t-1,t dbr fields
    bp_filter = filter_gaussian((n_rows, n_cols), n_levels)
    rad_cascades = []
    for ia in range(3):
        cascade = decomposition_fft(db_fields[ia, :, :], bp_filter,
                                    normalize=True, compute_stats=True)
        rad_cascades.append(cascade)

    # calculate the parameters
    params = calculate_parameters(
        db_fields[2, :, :], rad_cascades, oflow, scale_break, zero_value, minutes)

    params.set_metadata("valid_time", base_time)
    params.set_metadata("domain", dname)
    params.set_metadata("product", "qpe")

    # Set up the valid_times for the forecasts and the ensemble numbers 
    valid_times =  [base_time+ia*dt for ia in range(1, n_forecasts+1)] 
    ensembles = [ia for ia in range(n_ensembles)] 

    # Make the ensemble forecasts
    # Preallocate output: [ens, t, y, x]
    ens_stack = np.empty((n_ensembles, n_forecasts, n_rows, n_cols), dtype=np.float32)
    for ens in range(n_ensembles):
        fx_cascades = [
            copy.copy(rad_cascades[2]), # t0
            copy.copy(rad_cascades[1]), # t-1
        ]

        for ix in range(n_forecasts):
            fx_dbrain = update_field(
                fx_cascades, oflow, params, bp_filter, psize, scale_break_km, db_threshold, zero_value)
            fx_rain = transformer.inverse_transform(fx_dbrain)
            ens_stack[ens, ix, :, :] = fx_rain.astype(np.float32, copy=False) 

            fx_cascades[1] = fx_cascades[0]
            fx_cascades[0] = decomposition_fft(
                fx_dbrain, bp_filter, compute_stats=True, normalize=True)

    # Make the output file name and geo-referencing dictionary 
    file_path = Path(make_nc_name(dname, mname, base_time)) 
    geo_mdata = generate_geo_dict(domain)

    # Write the data to the nc file 
    write_netcdf(file_path,ens_stack,geo_mdata,valid_times,ensembles)
if __name__ == "__main__":
    main()
