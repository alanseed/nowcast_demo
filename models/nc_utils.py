import numpy as np

def replace_extension(filename: str, new_ext: str) -> str:
    return f"{filename.rsplit('.', 1)[0]}{new_ext}"

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
    out_geo["cartesian_unit"] = 'm'
    out_geo["yorigin"] = 'lower'
    out_geo["unit"] = 'mm/h'
    out_geo["threshold"] = 0
    out_geo["transform"] = None

    return out_geo
