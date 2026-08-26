import os
from pathlib import Path
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import pandas as pd
from dotenv import load_dotenv
import geopandas as gpd
import coastpy
import pystac
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from skimage import measure
from pyproj import CRS
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree
import xarray as xr
import rioxarray as rxr
import contextily as ctx
from statsmodels.nonparametric.smoothers_lowess import lowess
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import yaml

def subsidence_translation(da,year,global_rate=None,region=None,raster_path=None,data_fill='median'):

    def fill_nan_nearest(da):
        """Fill NaN cells using nearest valid neighbour."""
        arr = da.values
        mask = np.isnan(arr)

        if not mask.any():
            return da

        if mask.all():
            raise ValueError("All values are NaN; nearest filling is impossible.")

        idx = distance_transform_edt(
            mask,
            return_distances=False,
            return_indices=True
        )

        filled = arr[tuple(idx)]
        return da.copy(data=filled)


    if global_rate is not None:
        subsidence = global_rate / 100 * year  # cm/yr -> m total

        dep = da.dep if hasattr(da, "dep") else da
        dep_arr = dep.values.astype(float).copy()

        nodata = dep.rio.nodata
        print(np.isfinite(nodata),'aa',nodata)
        nodata = dep.rio.nodata
        if nodata is None or not np.isfinite(nodata):
            nodata = dep.attrs.get("_FillValue", -9999.0)

        # valid DEM cells only
        valid = np.isfinite(dep_arr) & (dep_arr != nodata)

        # apply subsidence only to valid cells
        new_dep = dep_arr.copy()
        new_dep[valid] = dep_arr[valid] - subsidence

        # keep nodata as nodata, not NaN
        new_dep[~valid] = np.nan

        return new_dep
    
    elif region is not None:
        # -----------------------------
        # Check input
        # -----------------------------
        if raster_path is None:
            raise ValueError('specify path for subsidence raster')
        # raster_path = r"f:\TUD_CEG_Repository\MSc_Thesis\Data\Subsidence\Final_subsidence_prediction.tif"

        # -----------------------------
        # Read data
        # -----------------------------
        print('Read subsidence value from data..')
        src = rxr.open_rasterio(raster_path).squeeze()
        # region = gpd.read_file(geom_files)
        # region = geom_files
        raster_crs = src.rio.crs
        target_crs = region.crs

        # -----------------------------
        # Clip raster to region bbox
        # -----------------------------
        print('Clip raster input from Hasan...')
        region_raster_crs = region.to_crs(raster_crs)

        minx, miny, maxx, maxy = region_raster_crs.total_bounds

        subset = src.rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy
        )

        # convert nodata to NaN
        if subset.rio.nodata is not None:
            subset = subset.where(subset != subset.rio.nodata)


        # -----------------------------
        # Reproject to target/grid CRS
        # -----------------------------
        subset = subset.rio.reproject(target_crs)

        # -----------------------------
        # Fill missing data with statistics
        # -----------------------------
        print('Fill missing data...')
        if data_fill == 'median':
            data_fill = subset.median()
            subset_filled = subset.fillna(data_fill)
        elif isinstance(data_fill,float):
            subset_filled = subset.fillna(data_fill)
        # -----------------------------
        # Interpolate raster to model grid
        # -----------------------------
        subset_interp = subset_filled.interp(
            x=xr.DataArray(da["xc"].values, dims=("y", "x")),
            y=xr.DataArray(da["yc"].values, dims=("y", "x")),
            method="nearest"
        )

        # -----------------------------
        # Fill edge NaNs using nearest valid value
        # -----------------------------
        da_filled = fill_nan_nearest(subset_interp)
        nodata = subset.rio.nodata
        dz = da_filled.values.copy()
        dz = np.where((dz) == 1, 1, dz)
        dz = np.where((dz) == 5, 3, dz)
        dz = np.where((dz) == 10, 5, dz)
        dz = np.where((dz) == -9999, np.nan, dz)
        subsidence = dz/100 * year
        new_dep = da - subsidence

        return subsidence, new_dep.dep.values
    
    else:
        return print('No subsidence data given')


def shoreline_translation(da,res,gdf_roi,years=100,cl_file=None,normal_direction=1,check_data=True,resample=100,
                          user_buffer=None,building=None):
    """ 
    da = DEM, xarray
    res = grid resolution, int
    gdf = geodataframe containing change rate, geometry is transect
    years = projection year, int
    cl_file = manual defined initial shoreline, geojson
    dem_crs = crs of the dem, int
    """

    # get the transform matrix for the grid coordinate
    crs_wkt = da.rio.crs
    A, b, A_inv = affine_from_xc_yc(da.xc, da.yc)


    # check input type of shoreline file
    if isinstance(cl_file, Path):
        cl_utm = gpd.read_file(cl_file).to_crs(crs_wkt)
        cl_grid = utm2grid(cl_utm,b,A_inv)
        print('Getting coastline from file...')
    elif cl_file == 'gctr':
        print('Getting coastline from GCTR...')
        centroids = gdf_roi.sort_values("transect_id").to_crs(crs_wkt).geometry.centroid

        line = LineString([(p.x, p.y) for p in centroids])

        cl_utm = gpd.GeoDataFrame(
            {"name": ["centroid_shoreline"]},
            geometry=[line],
            crs=crs_wkt
        )
        cl_grid = utm2grid(cl_utm,b,A_inv)
    elif cl_file == 'zero':
        print('Getting coastline from contour')
        cl_utm, cl_grid = extract_contour_from_dem(xc=da["xc"].values,
                                            yc=da["yc"].values,
                                            z=da.dep.values,
                                            level=0.0,
                                            crs=crs_wkt,#"EPSG:32749",
                                            min_length=5000)  # meters
    else:
        raise 'Error shoreline format'


    print('Initial shoreline defined...')

    step_m = resample
    s_samp, xy_samp = resample_line(cl_utm, step_m) # return distance (0,step,...,line length) and xy coordinate for those points
    xy_samp = xy_samp.squeeze()

    print('Shoreline resampled...')
    n_samp = normals_from_xy(xy_samp) # vector of normal coordinate for each point
    n_samp = n_samp*normal_direction
    coast_line = LineString(cl_utm.get_coordinates().values)




    print('Applied change rate to resampled shoreline...')

    r_smooth, r_raw, _ = clean_smooth_rate_alongshore(
        coast_line=coast_line,
        gdf_rate=gdf_roi.to_crs(crs_wkt),
        rate_col="sds:change_rate",
        s_query=s_samp.astype(float),
        max_snap=500,
        outlier_window=1000,
        outlier_thresh=3.5,
        smooth_window=21
    )

    dy_interp = r_smooth * years

    # change to true for inspecting raw coastline and refined coastline
    if False:
        return r_smooth, r_raw

    if isinstance(building,Path):
        print('Apply building erosion limit...')

        limit_gdf = gpd.read_file(building).to_crs(crs_wkt)

        dy_interp = limit_shift_by_barrier(
            xy_samp=xy_samp,
            normals=n_samp,
            dy=dy_interp,
            barrier_gdf=limit_gdf,
            safety_distance=2.0,
        )


    xy_shift = xy_samp + dy_interp[:,None] * n_samp  # "left" normal; flip sign if wrong side # move xy with dy magnitude and n_samp direction

    print('Project new shoreline...')
    shifted_cl_utm = LineString(xy_shift)


    print('Inspect new shoreline...')


    if check_data:
        check_shoreline_translation(
            cl_utm,
            shifted_cl_utm,
            s_samp=s_samp,
            n_samp=n_samp,
            dy_interp=dy_interp,
            xy_samp=xy_samp,
            xy_shift=xy_shift,
        )
    

    # change to true for inspecting the data with qgis
    if False:
        cl_utm.to_file(
            "original_shoreline_mozambique_short.geojson",
            driver="GeoJSON"
        )

        gpd.GeoDataFrame(
            geometry=[shifted_cl_utm],
            crs=cl_utm.crs
        ).to_file(
            "shifted_shoreline_mozambique_short.geojson",
            driver="GeoJSON"
        )

        return
    # assert 1 == 0

    

 

    n, m = da.dep.shape
    coast_mask = np.zeros((n,m), dtype=bool)

    # mark nearest integer pixels along the contour
    ii = np.clip(np.rint(cl_grid[:,0]).astype(int), 0, n-1)
    jj = np.clip(np.rint(cl_grid[:,1]).astype(int), 0, m-1)
    coast_mask[ii, jj] = True

    # distance (meters) to nearest coastline pixel
    dist_m = distance_transform_edt(~coast_mask, sampling=(res, res)) # calculate distance of nearest coastline for each grid

    # buffer_width

    max_shift = np.nanmax(np.abs(dy_interp))


    if user_buffer is not None:
        buffer_width = user_buffer
    else:
        buffer_width = max(
        2.0 * max_shift,   # shift-based
        4.0 * res,         # grid-size-based
        300.0,              # minimum physical width
    )


    print(f'buffer width = {buffer_width} m')

    in_buffer = dist_m <= buffer_width

    # Smooth decay weight (cosine taper is nice)
    w = np.zeros_like(dist_m, float)
    d = dist_m[in_buffer]
    w[in_buffer] = 0.5 * (1 + np.cos(np.pi * d / buffer_width))  # 1 at coast -> 0 at edge
    

    if check_data:
        fig, ax = plt.subplots(figsize=(9, 9))

        ax.scatter(
            da.xc.values,
            da.yc.values,
            s=2,
            alpha=0.2,
            label="domain cells",
            color='lightgrey'
        )

        ax.scatter(
            da.xc.values[in_buffer],
            da.yc.values[in_buffer],
            s=2,
            alpha=0.5,
            label="buffer grid cells",
            color='orange'
        )

        ax.plot(xy_samp[:, 0], xy_samp[:, 1], color='k', linewidth=2, label="original shoreline")
        ax.plot(xy_shift[:, 0], xy_shift[:, 1], linewidth=2, color='red', linestyle="--", label="shifted shoreline")

        ax.set_aspect("equal")
        ax.legend()
        plt.show()


    tree = cKDTree(xy_samp)  # coastline sample points

    # grid cell physical coords (flatten only inside buffer)
    I, J = np.nonzero(in_buffer) #convert 2d mask into list of active grid cells
    xg = da.xc.values[I, J]
    yg = da.yc.values[I, J]

    # nearest coastline sample per grid cell
    dist_to_sample, idx_near = tree.query(np.column_stack([xg, yg]), k=1)

    # displacement magnitude with decay weight
    shift = dy_interp[idx_near] * w[I, J]          # meters
    nxv = n_samp[idx_near, 0]
    nyv = n_samp[idx_near, 1]

    print("max dy:", np.abs(dy_interp).max())
    print("max shift:", np.abs(shift).max())
    print("max shift/res:", np.nanmax(np.abs(shift)) / res)
    print("mean shift/res:", np.nanmean(np.abs(shift)) / res)
    print("max w:", w.max())

    u_x = np.zeros_like(da.dep, float)
    u_y = np.zeros_like(da.dep, float)
    u_x[I, J] = shift * nxv  # meters
    u_y[I, J] = shift * nyv


    # Convert u(x,y) -> (dj,di)
    U = np.stack([u_x, u_y], axis=0).reshape(2, -1)  # (2, n*m)
    dJI = (A_inv @ U).reshape(2, *da.dep.shape)         # (2, n, m)
    dj = dJI[0]
    di = dJI[1]

    # Build sampling coordinates (backward mapping)
    ii, jj = np.indices(da.dep.shape)
    ii_src = ii - di
    jj_src = jj - dj


    # revise with this block
    dep0 = da.dep.values.astype(float).copy()

    nodata = da.dep.rio.nodata
    if nodata is None or not np.isfinite(nodata):
        nodata = -9999.0

    dep0[dep0 == nodata] = np.nan

    valid0 = np.isfinite(dep0).astype(float)
    dep_fill = np.where(np.isfinite(dep0), dep0, 0.0)

    warped_sum = map_coordinates(
        dep_fill,
        [ii_src, jj_src],
        order=1,
        mode="constant",
        cval=0.0
    )

    warped_weight = map_coordinates(
        valid0,
        [ii_src, jj_src],
        order=1,
        mode="constant",
        cval=0.0
    )

    warped = np.where(
        warped_weight > 0.99,
        warped_sum / warped_weight,
        np.nan
    )

    dep_new = dep0.copy()
    dep_new[in_buffer] = warped[in_buffer]

    # convert NaN back to SFINCS-compatible nodata
    dep_new = np.where(np.isfinite(dep_new), dep_new, np.nan)

    debug_variable = [w,u_x,u_y]


    return dep_new


def check_shoreline_translation(
    old_cl,
    new_cl,
    s_samp=None,
    n_samp=None,
    dy_interp=None,
    xy_samp=None,
    xy_shift=None,
    ):

    # New shifted coastline
    new_cl = gpd.GeoDataFrame(
        {"name": ["shifted"]},
        geometry=[new_cl],
        crs=old_cl.crs   # important: use same CRS as old_cl
    )

    # Reproject both to Web Mercator for satellite tiles
    old_3857 = old_cl.to_crs(epsg=3857)
    new_3857 = new_cl.to_crs(epsg=3857)

    # fig, ax = plt.subplots(figsize=(10, 10))

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    # fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    ax = axs[0]
    old_3857.plot(ax=ax, color="yellow", linewidth=2.5, label="Original coastline")
    new_3857.plot(ax=ax, color="red", linewidth=2.5, label="Shifted coastline")

    # Add satellite basemap

    ctx.add_basemap(
        ax=ax,
        source=ctx.providers.Esri.WorldImagery,
        zoom=12
    )

    ax.legend()
    ax.set_axis_off()
    ax.set_title("Original vs shifted coastline")
    # plt.show()

    # -------------------------
    # 2. Alongshore displacement
    # -------------------------
    ax = axs[1]

    if s_samp is not None and dy_interp is not None:
        print('second plot')
        ax.plot(s_samp / 1000, dy_interp, color="black", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)

        ax.set_xlabel("Alongshore distance, s (km)")
        ax.set_ylabel("Applied dy / normal displacement (m)")
        ax.set_title("Applied shoreline displacement")
        ax.grid(True, alpha=0.3)

        txt = (
            f"min dy = {np.nanmin(dy_interp):.2f} m\n"
            f"max dy = {np.nanmax(dy_interp):.2f} m\n"
            f"mean dy = {np.nanmean(dy_interp):.2f} m\n"
            f"median dy = {np.nanmedian(dy_interp):.2f} m"
        )

        ax.text(
            0.02, 0.15, txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )

    else:
        ax.text(0.5, 0.5, "No dy/s data provided", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()

    # -------------------------
    # 3. Distance check
    # -------------------------
    if xy_samp is not None and xy_shift is not None:
        dist = np.linalg.norm(xy_shift - xy_samp, axis=1)

        print("=== Shoreline displacement distance check ===")
        print(f"min distance  : {np.nanmin(dist):.3f} m")
        print(f"max distance  : {np.nanmax(dist):.3f} m")
        print(f"mean distance : {np.nanmean(dist):.3f} m")
        print(f"median distance: {np.nanmedian(dist):.3f} m")

        if dy_interp is not None:
            print("")
            print("=== dy check ===")
            print(f"min dy  : {np.nanmin(dy_interp):.3f} m")
            print(f"max dy  : {np.nanmax(dy_interp):.3f} m")
            print(f"mean dy : {np.nanmean(dy_interp):.3f} m")
            print(f"median dy: {np.nanmedian(dy_interp):.3f} m")
            print("")
            print("max |distance - abs(dy)|:",
                  np.nanmax(np.abs(dist - np.abs(dy_interp))))
            

    plt.show()


def limit_shift_by_barrier(
    xy_samp,
    normals,
    dy,
    barrier_gdf,
    crs=None,
    safety_distance=0.0,
):
    """
    xy_samp : (N, 2) original shoreline points
    normals : (N, 2) unit normal vectors
    dy      : (N,) proposed shoreline shift in meters
    barrier_gdf : building line/polygon movement limit
    safety_distance : optional setback before barrier, e.g. 5 m
    """

    barrier = barrier_gdf.geometry.union_all()

    dy_limited = dy.copy().astype(float)
    xy_shift_limited = xy_samp.copy().astype(float)

    for i, (p0_xy, nvec, d) in enumerate(zip(xy_samp, normals, dy)):
        if not np.isfinite(d):
            dy_limited[i] = np.nan
            continue

        p0 = Point(p0_xy)
        p1_xy = p0_xy + d * nvec
        p1 = Point(p1_xy)

        move_line = LineString([p0, p1])

        # if movement path does not hit barrier, keep original shift
        if not move_line.intersects(barrier):
            xy_shift_limited[i] = p1_xy
            continue

        inter = move_line.intersection(barrier)

        # collect intersection points
        pts = []

        if inter.geom_type == "Point":
            pts = [inter]
        elif inter.geom_type == "MultiPoint":
            pts = list(inter.geoms)
        elif inter.geom_type in ["LineString", "MultiLineString", "GeometryCollection"]:
            for geom in getattr(inter, "geoms", [inter]):
                if geom.geom_type == "Point":
                    pts.append(geom)
                elif geom.geom_type == "LineString":
                    pts.append(Point(geom.coords[0]))

        if len(pts) == 0:
            continue

        # choose nearest intersection along movement direction
        distances = np.array([p0.distance(pt) for pt in pts])
        d_hit = distances.min()

        # keep same sign as original dy
        d_allowed = max(0, d_hit - safety_distance)
        dy_limited[i] = np.sign(d) * min(abs(d), d_allowed)


    return dy_limited

def clean_smooth_rate_alongshore(
    coast_line,
    gdf_rate,
    rate_col,
    s_query,
    max_snap=500,
    outlier_window=1000,
    outlier_thresh=3.5,
    smooth_window=21
):

    print('Assign change rate to resampled shoreline...')
    s_src, r_src, r_raw = map_data_to_chainage(coast_line,gdf_rate,rate_col,max_snap)

    # 2. MAD outlier removal

    def remove_outliers_mad(x, thresh=3.5):
        x = np.asarray(x)

        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))

        if mad == 0:
            return np.ones(len(x), dtype=bool)

        z = 0.6745 * (x - med) / mad

        return np.abs(z) < thresh

    mask = remove_outliers_mad(r_src, thresh=3.5)

    s_clean = s_src[mask]
    r_clean = r_src[mask]

    print('Outlier removed....')

 

    # 3. interpolate removed points back to uniform query chainage
    r_raw = np.interp(s_query, s_src,r_raw)
    r_mad = np.interp(s_query, s_src, r_src)
    r_interp = np.interp(s_query, s_clean, r_clean)

    # 4. smooth alongshore signal


    r_smooth = (
        pd.Series(r_interp)
        .rolling(smooth_window, center=True, min_periods=1)
        .median()
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    return r_smooth, r_raw, r_mad

def utm2grid(cl_utm,b,A_inv):
       
    def xy_to_ij(x, y):
        xy = np.stack([x, y], axis=-1)
        ji = (xy - b) @ A_inv.T
        j = ji[..., 0]
        i = ji[..., 1]
        return i, j

    # 3. Convert one LineString from x/y to row/col
    def line_xy_to_ij(line):
        coords = np.array(line.coords)
        x = coords[:, 0]
        y = coords[:, 1]

        i, j = xy_to_ij(x, y)

        # grid coordinate line: x = column j, y = row i
        return LineString(np.column_stack([j, i]))

    # 4. Apply to GeoDataFrame
    def geom_xy_to_ij(geom):
        if geom.geom_type == "LineString":
            return line_xy_to_ij(geom)

        elif geom.geom_type == "MultiLineString":
            return MultiLineString([line_xy_to_ij(g) for g in geom.geoms])

        else:
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    gdf_grid = cl_utm.copy()
    gdf_grid["geometry"] = gdf_grid.geometry.apply(geom_xy_to_ij)
    gdf_grid = gdf_grid.set_crs(None, allow_override=True)


    coords = np.asarray(gdf_grid.get_coordinates())

    cl_grid = np.column_stack([
        coords[:,1],   # i
        coords[:,0],   # j
    ])

    return cl_grid

def map_data_to_chainage(coast_line,gdf_rate,rate_col,max_snap):
    # snap/project rate transects or points to shoreline chainage
    d = gdf_rate.geometry.distance(coast_line)
    gdf = gdf_rate.loc[d <= max_snap].copy()


    s_src = gdf.geometry.centroid.apply(coast_line.project).to_numpy()
    r_src = gdf[rate_col].to_numpy(dtype=float)

    valid = np.isfinite(s_src) & np.isfinite(r_src)
    s_src = s_src[valid]
    r_src = r_src[valid]

    order = np.argsort(s_src)
    s_src = s_src[order]
    r_src = r_src[order]
    r_raw = r_src
    

    return s_src, r_src, r_raw


def build_dy_of_s(coast_line, gdf_rate, rate_col, years, max_snap=None):
    # Optional: ignore points too far from coastline
    d = gdf_rate.geometry.distance(coast_line)
    if max_snap is not None:
        gdf_rate = gdf_rate.loc[d <= max_snap].copy()

    s_src = gdf_rate.geometry.centroid.apply(coast_line.project).to_numpy() # convert GCTTR coor to distance on extracted shoreline
    # s_src = gdf_rate.geometry.to_numpy()
    r_src = gdf_rate[rate_col].to_numpy(dtype=float)

    order = np.argsort(s_src)
    s_src = s_src[order]
    r_src = r_src[order]

    r_smooth = lowess(
    r_src,
    s_src,
    frac=0.1,      # 10% of coastline length
    return_sorted=False
)

    def dy(s_query): # interpolate the rate from shoreline distance
        # r = np.interp(s_query, s_src, r_src)  # m/year
        r = np.interp(s_query, s_src, r_smooth)
        return r * years                      # meters

    return dy


def resample_line(line, step_m):
    L = line.length.values
    s = np.arange(0, L + step_m, step_m)
    pts = [line.interpolate(si) for si in s]
    xy = np.array([(p.x, p.y) for p in pts], dtype=float)
    return s, xy

def normals_from_xy(xy):
    d = np.zeros_like(xy)
    d[1:-1] = xy[2:] - xy[:-2]
    d[0] = xy[1] - xy[0]
    d[-1] = xy[-1] - xy[-2]
    tnorm = np.linalg.norm(d, axis=1)
    tnorm[tnorm == 0] = 1.0
    t = d / tnorm[:,None]
    n_left = np.column_stack([-t[:,1], t[:,0]])
    return n_left

def affine_from_xc_yc(xc, yc):
    # Use 3 points to solve for A,b robustly using least squares
    # Model: x = a*j + b*i + c ; y = d*j + e*i + f
    n, m = xc.shape
    pts = [(0,0), (0,m-1), (n-1,0), (n-1,m-1), (n//2, m//2)]
    M = []
    vx = []
    vy = []
    for i,j in pts:
        M.append([j, i, 1.0])
        vx.append(xc[i,j])
        vy.append(yc[i,j])
    M = np.array(M, float)
    vx = np.array(vx, float)
    vy = np.array(vy, float)

    ax, bx, cx = np.linalg.lstsq(M, vx, rcond=None)[0]
    ay, by, cy = np.linalg.lstsq(M, vy, rcond=None)[0]

    A = np.array([[ax, bx],
                  [ay, by]], dtype=float)
    b = np.array([cx, cy], dtype=float)

    A_inv = np.linalg.inv(A)
    return A, b, A_inv


def extract_contour_from_dem(xc, yc, z, level, crs="EPSG:4326"):
    """
    xc, yc: 2D arrays (ny, nx) with coordinates
    z:      2D array (ny, nx)
    level:  contour level
    """
    ny, nx = z.shape
    contours = measure.find_contours(z, level=level)  # returns (row, col) floats

    lines = []
    for c in contours:
        row = c[:, 0]
        col = c[:, 1]

        # bilinear sample xc,yc at fractional indices
        r0 = np.floor(row).astype(int)
        c0 = np.floor(col).astype(int)
        r1 = np.clip(r0 + 1, 0, ny - 1)
        c1 = np.clip(c0 + 1, 0, nx - 1)

        dr = row - r0
        dc = col - c0

        x00 = xc[r0, c0]; x10 = xc[r1, c0]; x01 = xc[r0, c1]; x11 = xc[r1, c1]
        y00 = yc[r0, c0]; y10 = yc[r1, c0]; y01 = yc[r0, c1]; y11 = yc[r1, c1]

        x = (1-dr)*(1-dc)*x00 + dr*(1-dc)*x10 + (1-dr)*dc*x01 + dr*dc*x11
        y = (1-dr)*(1-dc)*y00 + dr*(1-dc)*y10 + (1-dr)*dc*y01 + dr*dc*y11

        coords = np.column_stack([x, y])
        if coords.shape[0] >= 2:
            line = LineString(coords)
            lines.append(line)

    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)

    idx = gdf.length.idxmax()
    cl_utm = gdf.loc[[idx]].copy()
    cl_grid = contours[idx]

    return cl_utm, cl_grid


def DEM_resampling(src_tif,out_dir,location,scale=10):

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f'{location}_10m.tif'


    with rasterio.open(src_tif) as src:
        data = src.read(
            out_shape=(
                src.count,
                src.height * scale,
                src.width * scale,
            ),
            resampling=Resampling.bilinear,
        )

        # update transform for smaller pixels
        transform = src.transform * Affine.scale(
            src.width / data.shape[-1],
            src.height / data.shape[-2],
        )

        profile = src.profile.copy()
        profile.update(
            height=data.shape[-2],
            width=data.shape[-1],
            transform=transform,
            compress="lzw",
            dtype=data.dtype,
            nodata=src.nodata,
        )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data)


def add_raster_to_yml(
    yml_path,
    dataset_name,
    file_path,
    crs,
):
    # Read existing YAML
    with open(yml_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f) or {}

    # Define new/updated entry
    catalog[dataset_name] = {
        "data_type": "RasterDataset",
        "uri": str(file_path),
        "driver": {
            "name": "rasterio",
            "options": {
                "chunks": {
                    "x": 1000,
                    "y": 1000,
                }
            },
        },
        "metadata": {
            "category": "topography",
            "crs": str(crs),
            "nodata": -9999,
        },
        "data_adapter": {
            "rename": {
                "large_model_dep": "elevtn"
            }
        },
    }

    # Write updated catalog
    with open(yml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            catalog,
            f,
            sort_keys=False,
        )

def change_yml_root(yml_path, new_root):
    new_root = str(new_root)

    with open(yml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(yml_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("root:"):
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}root: {new_root}\n"

            f.write(line)

def modify_DEM(da, region,
               erosion=False,subsidence=False,
               gctr_file = None,
               coastline_file = None,
               retreat_limit_file = None,
               performed_manual_edit = False,
               buffer_width = None,
               normal_direction = 1,
               subsidence_file = None,
               nyear=0,
               resolution=10,resample=100,
               ):
    
    dep_old = np.copy(da.dep.values)

    if erosion:
        # read modified gctr  
        gctr = gpd.read_parquet(gctr_file)
        # modify manually added new structure attribute to filter area (optional)
        if performed_manual_edit:
            gctr['initial_rate'] = gctr['sds:change_rate']
            mask = gctr["structure"] == True
            gctr.loc[mask, "sds:change_rate"] = 0

        horizontal_shift = shoreline_translation(da,resolution,gctr,years=nyear,
                                                 cl_file=coastline_file,normal_direction=normal_direction,
                                                 check_data=False,resample=resample,
                                                 user_buffer=buffer_width,building=retreat_limit_file)
        
        da['dep'].values = horizontal_shift
        dep_new = np.copy(horizontal_shift)

    if subsidence:
        dz, zz = subsidence_translation(da,nyear,region=region,raster_path=subsidence_file)
        da['dep'].values = zz
        dep_new = np.copy(zz)


    if not (erosion & subsidence):
        dep_new = np.copy(da.dep.values)

    return dep_old, dep_new


