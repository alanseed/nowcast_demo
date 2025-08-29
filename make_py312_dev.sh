# 1) fresh env on Python 3.12
conda create -n pysteps_dev312 -y python=3.12
conda activate pysteps_dev312
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# 2) core scientific stack
conda install -y numpy scipy matplotlib xarray dask netcdf4 cython pip

# 3) geo stack from conda-forge (prebuilt & compatible)
conda install -y cartopy pyproj shapely proj proj-data geos

# 4) your editable pysteps
python -m pip install -e ~/alan/pysteps

# 5) sanity checks
python -c "import sys; print(sys.version); import cartopy, shapely, pyproj; \
print('cartopy:', cartopy.__version__); \
import pysteps; print('pysteps:', pysteps.__version__, '->', pysteps.__file__)"
