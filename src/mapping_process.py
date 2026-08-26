import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import Point, LineString, MultiLineString
import rasterio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import pystac
import coastpy
from dotenv import load_dotenv
import shutil


def initialize_gcts(env_path,columns,bbox = [-180,-90,180,90]):
    env_path = r'f:\TUD_CEG_Repository\Internship\Code\ATT64895.env'
    load_dotenv(env_path) # may need to change the path
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    storage_options = {"account_name": "coclico", "sas_token": sas_token}

    coclico_catalog = pystac.Catalog.from_file(
        "https://coclico.blob.core.windows.net/stac/v1/catalog.json"
    )
    collection = coclico_catalog.get_child("gctr")

    db = coastpy.io.STACQueryEngine(
        stac_collection=collection,
        storage_backend="azure",
        columns = columns  # when you don't need all data
    )

    west, east, south, north = bbox[0], bbox[2], bbox[1], bbox[3]
    gdf = db.get_data_within_bbox(west, south, east, north, sas_token=sas_token)

    return gdf

def build_valid_pixel_tree(src):
    """KDTree of valid pixel centers + their values (band 1)."""
    band1 = src.read(1, masked=True)
    mask = ~band1.mask
    rows, cols = np.where(mask)

    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
    
    # transformer = Transformer.from_crs(src.crs, 3857, always_xy=True)
    # xs, ys = transformer.transform(xs, ys)


    pts = np.column_stack([xs, ys]).astype("float64")
    vals = np.asarray(band1[rows, cols]).astype("float64")

    tree = cKDTree(pts)
    return tree, vals


def sampling_nearest_pixel_to_gcts(
    gcts,
    raster_path,
    value_col="subsidence",
    dist_col="dist_m",
    max_dist=None
):
    """
    Assign nearest valid raster pixel value to each line centroid.

    Parameters
    ----------
    gcts : GeoDataFrame
        Input lines.
    raster_path : str
        Raster file path.
    value_col : str
        Output column for raster value.
    dist_col : str
        Output column for distance to nearest valid pixel.
    max_dist : float or None
        Maximum allowed distance in raster CRS units.
        If nearest pixel is farther than this, value_col will be NaN.

    Returns
    -------
    GeoDataFrame
    """
    with rasterio.open(raster_path) as src:
        # Reproject lines to raster CRS
        gdf_r = gcts.to_crs(src.crs) if gcts.crs != src.crs else gcts.copy()

        # Build KDTree in raster CRS
        tree, vals = build_valid_pixel_tree(src)

        # Centroids in raster CRS
        centroids = gdf_r.to_crs(3857).geometry.centroid.to_crs(4326)
        xy = np.column_stack([centroids.x.values, centroids.y.values]).astype("float64")

        # Query nearest valid pixel
        dist, idx = tree.query(xy, k=1)

        # Extract values
        nearest_vals = vals[idx].astype("float64")

        # Apply user-specified maximum distance
        if max_dist is not None:
            nearest_vals = np.where(dist <= max_dist, nearest_vals, np.nan)

        # Assign back to original gdf
        gdf_out = gcts.copy()
        gdf_out[value_col] = nearest_vals
        gdf_out[dist_col] = dist

        return gdf_out


def _iter_lines(geom):
    """Yield LineString parts from LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield g


def _sample_line(line, src, step, n = 40):
    """
    Sample raster band 1 along a LineString.
    Returns distances, xs, ys, values (nodata -> nan).
    """
    # line = line

    length = line.length
    if length <= 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    dx = length/n
    distances = np.arange(0, length + dx, dx)
    points = [line.interpolate(d) for d in distances]

    xs = np.array([p.x for p in points], dtype="float64")
    ys = np.array([p.y for p in points], dtype="float64")
    coords = list(zip(xs, ys))

    values = np.array([v[0] for v in src.sample(coords)], dtype="float64")

    if src.nodata is not None:
        values[values == src.nodata] = np.nan

    return distances, xs, ys, values


def assign_raster_stats(
    gdf_lines: gpd.GeoDataFrame,
    raster_path: str,
    step: float | None = None,
    prefix: str = "rast",
):
    """
    Extract raster stats along each line.

    Returns:
        gdf_out  -> original GeoDataFrame with stats columns
    """

    with rasterio.open(raster_path) as src:

        # --- Reproject to raster CRS ---
        gdf = gdf_lines.to_crs(src.crs) if gdf_lines.crs != src.crs else gdf_lines.copy()

        # --- Auto step = pixel resolution if not given ---
        if step is None:
            resx, resy = src.res
            step = float((abs(resx) + abs(resy)) / 2)

        # --- Prepare outputs ---
        out_mean, out_max, out_min, out_p90, out_n = [], [], [], [], []
        # profile_rows = []

        # --- Loop through each line ---
        n = 0
        for idx, geom in zip(gdf_lines['transect_id'], gdf.geometry):

            # n = n + 1

            # if n%1000 == 0:
            #     print(f'{n}/{len(gdf)}')
            #     return

            all_vals = []

            for _, line in enumerate(_iter_lines(geom)):

                _, _, _, vals = _sample_line(line, src, step)

                if vals.size == 0:
                    continue

                all_vals.append(vals)

            # --- Summary statistics ---
            if not all_vals:
                out_mean.append(np.nan)
                out_max.append(np.nan)
                out_min.append(np.nan)
                out_p90.append(np.nan)
                out_n.append(0)
                continue

            v = np.concatenate(all_vals)
            v = v[np.isfinite(v)]

            if v.size == 0:
                out_mean.append(np.nan)
                out_max.append(np.nan)
                out_min.append(np.nan)
                out_p90.append(np.nan)
                out_n.append(0)
            else:
                out_mean.append(float(np.mean(v)))
                out_max.append(float(np.max(v)))
                out_min.append(float(np.min(v)))
                out_p90.append(float(np.percentile(v, 90)))
                out_n.append(int(v.size))

        # --- Attach stats back to original GDF ---
        gdf_out = gdf_lines.copy()
        gdf_out[f"{prefix}_mean"] = out_mean
        gdf_out[f"{prefix}_max"] = out_max
        gdf_out[f"{prefix}_min"] = out_min
        # gdf_out[f"{prefix}_p90"] = out_p90
        gdf_out[f"{prefix}_n"] = out_n

        return gdf_out
    

def sampling_intersect_pixel_to_gcts(
    gdf,
    raster_path,
    prefix="flood",
    batch_size=100_000,
    data_dir=Path.cwd(),
):

    import shutil
    temp_dir = Path(data_dir) / "temp"

    # Create temporary directory if it does not exist
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Process and save batches
    # --------------------------------------------------
    for batch, i in enumerate(range(0, len(gdf), batch_size)):

        gdf_batch = assign_raster_stats(
            gdf_lines=gdf.iloc[i:i + batch_size],
            raster_path=raster_path,
            prefix=prefix,
        )

        gdf_batch.to_parquet(
            temp_dir / f"hazard-{batch:06d}.parquet",
            engine="pyarrow",
            index=False,
            compression="snappy",
        )

    # --------------------------------------------------
    # Read and concatenate batches
    # --------------------------------------------------
    files = sorted(temp_dir.glob("hazard-*.parquet"))

    batches = []

    for hazard_file in files:
        batches.append(gpd.read_parquet(hazard_file))

    gdf_out = pd.concat(
        batches,
        ignore_index=True,
    )

    gdf_out = gpd.GeoDataFrame(
        gdf_out,
        geometry="geometry",
        crs=gdf.crs,
    )

    # --------------------------------------------------
    # Delete temporary directory and its contents
    # --------------------------------------------------
    shutil.rmtree(temp_dir)

    return gdf_out

