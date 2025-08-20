import numpy as np
import pandas as pd
import logging 

from pymongo.database import Database
from pymongo import ASCENDING
import datetime

from pysteps import motion, extrapolation
from pysteps.cascade.decomposition import decomposition_fft
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.utils import DBTransformer

from rainfields_db import write_state, get_state, make_nc_name, get_rainfield
from models.nc_utils import replace_extension 
from models.db_utils import ensure_utc, get_config

def lagr_auto_cor(data: np.ndarray, oflow: np.ndarray, config: dict):
    """
    Generate the Lagrangian auto correlations for STEPS cascades.

    Args:
        data (np.ndarray): [T, L, M, N] where:
            - T = ar_order + 1 (number of time steps)
            - L = number of cascade levels
            - M, N = spatial dimensions.
        oflow (np.ndarray): [2, M, N] Optical flow vectors.
        config (dict): Configuration dictionary containing:
            - "n_cascade_levels": Number of cascade levels (L).
            - "ar_order": Autoregressive order (1 or 2).
            - "extrapolation_method": Method for extrapolating fields.

    Returns:
        np.ndarray: Autocorrelation coefficients of shape (L, ar_order).
    """

    n_cascade_levels = config["n_cascade_levels"]
    ar_order = config["ar_order"]

    if data.shape[0] < (ar_order + 1):
        raise ValueError(
            f"Insufficient time steps. Expected at least {ar_order + 1}, got {data.shape[0]}.")

    extrapolation_method = extrapolation.get_method("semilagrangian")

    autocorrelation_coefficients = np.full(
        (n_cascade_levels, ar_order), np.nan)

    for level in range(n_cascade_levels):
        lag_1 = extrapolation_method(data[-2, level], oflow, 1)[0]
        lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

        data_t = np.where(np.isfinite(data[-1, level]), data[-1, level], 0)
        if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
            autocorrelation_coefficients[level, 0] = np.corrcoef(
                lag_1.flatten(), data_t.flatten())[0, 1]

        if ar_order == 2:
            lag_2 = extrapolation_method(data[-3, level], oflow, 1)[0]
            lag_2 = np.where(np.isfinite(lag_2), lag_2, 0)

            lag_1 = extrapolation_method(lag_2, oflow, 1)[0]
            lag_1 = np.where(np.isfinite(lag_1), lag_1, 0)

            if np.std(lag_1) > 1e-1 and np.std(data_t) > 1e-1:
                autocorrelation_coefficients[level, 1] = np.corrcoef(
                    lag_1.flatten(), data_t.flatten())[0, 1]

    return autocorrelation_coefficients


def calculate_wavelengths(n_levels: int, domain_size: float, d: float = 1.0):
    """
    Compute the central wavelengths (in km) for each cascade level.

    Parameters
    ----------
    n_levels : int
        Number of cascade levels.
    domain_size : int or float
        The larger of the two spatial dimensions of the domain in pixels.
    d : float
        Sample frequency in pixels per km. Default is 1.

    Returns
    -------
    wavelengths_km : np.ndarray
        Central wavelengths in km for each cascade level (length = n_levels).
    """
    # Compute q
    q = pow(0.5 * domain_size, 1.0 / n_levels)

    # Compute central wavenumbers (in grid units)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n_levels + 1)]
    central_wavenumbers = np.array([0.5 * (r0 + r1) for r0, r1 in r])

    # Convert to frequency
    central_freqs = central_wavenumbers / domain_size
    central_freqs[0] = 1.0/domain_size
    central_freqs[-1] = 0.5  # Nyquist limit

    # Convert wavelength to km, d is pixels per km 
    central_freqs = central_freqs * d
    central_wavelengths_km = 1.0 / central_freqs
    return central_wavelengths_km

def make_states_df(file_names: list[str], db: Database, config: dict, domain: dict):
    """
    Generate the cascade and motion vectors (oflow) for a list of filenames 
    Writes the state to the database if needed

    Args:
        file_names (List[str]): List of filenames to process
        db (Database): Database
        config (Dict): Configuration of the model
        domain (Dict): Configuration of the domain 

    Returns:
        _type_: DataFrame of the states 
    """
    timestep = config["timestep"]
    n_levels = config['n_cascade_levels']
    rain_threshold = config["precip_threshold"]
    n_rows = domain.get("n_rows")
    n_cols = domain.get("n_cols")

    def zero_oflow(): return np.zeros(
        (2, n_rows, n_cols), dtype=np.float32)  # type: ignore

    oflow_method = motion.get_method("LK")  # Lucas-Kanade method
    bp_filter = filter_gaussian((n_rows, n_cols), n_levels)
    db_transformer = DBTransformer(rain_threshold)

    # Initialize buffers for processing in time sequence
    states = []
    oflow = np.array([])

    for file_name in sorted(file_names):
        found_state = True
        rain_data, rain_metadata = get_rainfield(db, file_name)

        # Continue to next file if missing
        if rain_data.size == 0:
            continue

        db_data = db_transformer.transform(rain_data.to_numpy())
        db_metadata = db_transformer.get_metadata()

        # Check if cascade already exists for this file and make one if needed
        state_filename = replace_extension(file_name, ".npz")
        cascade_dict, oflow, state_metadata = get_state(db, state_filename)
        if not cascade_dict:
            found_state = False
            cascade_dict = decomposition_fft(
                db_data, bp_filter, compute_stats=True, normalize=True
            )

        # Add the rain field transformation for the cascade
        cascade_dict["transform"] = db_metadata["transform"]
        cascade_dict["zerovalue"] = db_metadata["zerovalue"]
        cascade_dict["threshold"] = db_metadata["threshold"]

        field_metadata = {
            "filename": state_filename,
            "domain": rain_metadata.get("domain"),
            "product": rain_metadata.get("product"),
            "valid_time": rain_metadata.get("valid_time"),
            "base_time": rain_metadata.get("base_time"),
            "ensemble": rain_metadata.get("ensemble"),
            "mean": rain_metadata.get("mean", 0),
            "std_dev": rain_metadata.get("std_dev", 0),
            "wetted_area_ratio": rain_metadata.get("wetted_area_ratio", 0)
        }
        valid_time = ensure_utc(field_metadata["valid_time"])
        base_time = ensure_utc(field_metadata["base_time"])

        # Compute optical flow if needed
        if oflow.size == 0:
            prev_time = valid_time - datetime.timedelta(seconds=timestep)
            p_filename = make_nc_name(field_metadata["domain"], field_metadata["product"],
                                      prev_time, base_time, field_metadata["ensemble"])
            prev_rain, p_metadata = get_rainfield(db, p_filename)
            if prev_rain.size == 0:
                # We have a cascade but cant do the tracking because of no rain
                oflow = zero_oflow()
            else:
                prev_field = db_transformer.transform(prev_rain.to_numpy())
                R = np.array([prev_field, db_data])
                oflow = oflow_method(R)

        # Write the state to the database if needed
        if not found_state:
            write_state(db, cascade_dict, oflow, state_filename, field_metadata)

        record = {
            "domain": field_metadata["domain"],
            "product": field_metadata["product"],
            "valid_time": valid_time,
            "base_time": base_time,
            "ensemble": field_metadata["ensemble"],
            "cascade": cascade_dict,
            "motion_field": oflow
        }
        states.append(record)

    states_df = pd.DataFrame(states)
    return states_df
