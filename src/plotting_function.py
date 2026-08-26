import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import geopandas as gpd
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorcet as cc
from shapely.geometry import box
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
    })


def initialize_map(figsize=(16,8),projection = ccrs.PlateCarree()):
    fig = plt.figure(figsize=figsize)

    projection = projection
    # projection = ccrs.Robinson()

    ax = plt.axes(projection=projection)
    
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#cacaca")
    ax.add_feature(cfeature.BORDERS,
                linewidth=0.3,
                edgecolor='white')

    ax.coastlines(linewidth=0.3, color='white')

    gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.2,
    color="gray",
    alpha=0.2
)

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    return fig, ax


    
def visualize_subsidence(
    df,
    key,
    bins=(0, 3, 7.5, 12),
    label="Subsidence rate (cm/year)",
    cmap=plt.cm.YlOrRd,
    base_value=1,
    base_markersize=4,
    high_markersize=45,
    figsize=(16, 8),
    projection=None,
):

    if projection is None:
        projection = ccrs.Robinson()

    if df.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS. Please set CRS first."
        )

    # Reproject once for plotting
    gdf = df.to_crs(4326)

    # Discrete color normalization
    norm = mcolors.BoundaryNorm(
        boundaries=bins,
        ncolors=cmap.N,
        clip=True
    )

    # Initialize common map
    fig, ax = initialize_map(
        figsize=figsize,
        projection=projection
    )

    # Separate base/low class from higher subsidence
    base_mask = gdf[key] == base_value
    high_mask = ~base_mask

    # Plot base class first
    if base_mask.any():
        gdf.loc[base_mask].plot(
            column=key,
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            markersize=base_markersize,
            alpha=0.45,
            legend=False,
        )

    # Plot higher classes on top
    if high_mask.any():
        gdf.loc[high_mask].plot(
            column=key,
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            markersize=high_markersize,
            alpha=0.9,
            legend=True,
            legend_kwds={
                "label": label,
                "orientation": "horizontal",
                "shrink": 0.7,
                "pad": 0.06,
            },
        )

    # Fallback when only the base class exists
    else:
        gdf.plot(
            column=key,
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            markersize=base_markersize,
            alpha=0.7,
            legend=True,
            legend_kwds={
                "label": label,
                "orientation": "horizontal",
                "shrink": 0.7,
                "pad": 0.06,
            },
        )

    # Customize colorbar labels
    cbar_ax = fig.axes[-1]

    class_centers = [
        (bins[0] + bins[1]) / 2,
        (bins[1] + bins[2]) / 2,
        (bins[2] + bins[3]) / 2,
    ]

    cbar_ax.set_xticks(class_centers)
    cbar_ax.set_xticklabels(
        ["<1", "1–5", ">5"]
    )

    return fig, ax


def visualize_hazard(
    df,
    key,
    bins=[0, 1, 2, 3, 4, 5, 6],
    label="",
    cmap=plt.cm.turbo,
    figsize=(16, 8),
    projection=ccrs.Robinson()
):
    """
    Visualize a hazard variable on a global map.
    """

    # Discrete color normalization
    norm = mcolors.BoundaryNorm(
        bins,
        ncolors=cmap.N,
        clip=False
    )

    # Initialize map
    fig, ax = initialize_map(
        figsize=figsize,
        projection=projection
    )

    # Plot hazard data
    df.to_crs(4326).plot(
        column=key,
        cmap=cmap,
        ax=ax,
        norm=norm,
        transform=ccrs.PlateCarree(),
        legend=True,
        linewidth=2,
        legend_kwds={
            "label": label,
            "orientation": "horizontal",
            "shrink": 0.7,
            "pad": 0.06
        }
    )

    # Colorbar axis
    cbar_ax = fig.axes[-1]

    # Remove uppermost tick
    ticks = cbar_ax.get_xticks()

    if len(ticks) > 1:
        cbar_ax.set_xticks(ticks[:-1])

    return fig, ax


def ecdf(df,key,
         label=''):
    x = df[key].dropna().values
    x_sorted = np.sort(x)

    y = np.arange(1, len(x_sorted)+1) / len(x_sorted)


    return x_sorted, y


def inset_map(df,loc,label='',
              flood_extreme=2.65,erosion_extreme=1.01,subsidence_extreme=1):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    countries_shp = r'f:\TUD_CEG_Repository\MSc_Thesis\Data\Natural Earth\ne_110m_admin_0_countries.shp'
    world = gpd.read_file(countries_shp).to_crs(4326) 

    coast = df[df['h3'] == loc].copy()
    
    valid = coast.dropna(subset=['flood_mean', 'erosion', 'subsidence'])

    F = valid['flood_mean'] >= flood_extreme
    E = valid['erosion'] >= erosion_extreme
    S = valid['subsidence'] >= subsidence_extreme
    overlap = valid[F & E & S]


    xmin, ymin, xmax, ymax = coast.total_bounds
    xc = (xmin+xmax)/2
    yc = (ymin+ymax)/2

    # -------------------------
    # Main figure
    # -------------------------

    fig,ax = initialize_map(figsize=(14,14))


    # main detailed map
    valid.plot(
        ax=ax,
        color='steelblue',
        # linewidth=20,
        markersize=500,
        transform=ccrs.PlateCarree(),
        label='Transects with 3 hazards'
    )

        # main detailed map
    overlap.plot(
        ax=ax,
        color='red',
        # linewidth=20,
        markersize=500,
        transform=ccrs.PlateCarree(),
        label='Extreme hazard overlap',
    )


    ax.set_xlim((xc-4), (xc+4))
    ax.set_ylim((yc-4), (yc+4))

    ax.set_title(label,fontsize=16)

    # -------------------------
    # Inset map
    # -------------------------
    axins = inset_axes(
        ax,
        width="30%",
        height="30%",
        loc="lower left",
        borderpad=2
    )

    # recreate inset with projection
    axins.remove()

    axins = fig.add_axes(
        # [0.13, 0.3, 0.22, 0.22],
        [0.65, 0.65, 0.2, 0.2],
        projection=ccrs.PlateCarree()
    )

    # # global/regional context
    world.plot(
        ax=axins,
        color='lightgray',
        edgecolor='white'
    )


    # broader regional extent
    axins.set_extent([
        xc-15, xc+15,
        yc-15, yc+15
    ])


    ax.set_aspect("auto")  # fills the same subplot area

    # -------------------------
    # Add red bbox
    # -------------------------
    bbox = gpd.GeoSeries(
        [box(xmin, ymin, xmax, ymax)],
        crs="EPSG:4326"
    )

    bbox.boundary.plot(
        ax=axins,
        edgecolor='red',
        linewidth=1.5
    )

    # clean inset
    axins.set_xticks([])
    axins.set_yticks([])
    
    return fig, ax

def plot_overlap_index(
    df, label='',
    flood_col="Flood", erosion_col="Erosion", subs_col="Subsidence",
    flood_extreme=2.65, erosion_extreme=1.01, subsidence_extreme=1
):
    dfp = df.reset_index(drop=True).copy()
    x = dfp.index

    fig, ax = plt.subplots(figsize=(12, 3))

    def intervals(mask):
        mask = mask.fillna(False).astype(bool)
        idx = x[mask].to_numpy()

        if len(idx) == 0:
            return []

        breaks = idx[1:] != idx[:-1] + 1
        starts = idx[np.r_[0, np.where(breaks)[0] + 1]]
        ends = idx[np.r_[np.where(breaks)[0], len(idx) - 1]]

        return [(s, e - s + 1) for s, e in zip(starts, ends)]

    # Overlap mask
    F = dfp["flood_mean"] >= flood_extreme
    E = dfp["erosion"] >= erosion_extreme
    S = dfp["subsidence"] >= subsidence_extreme
    overlap_mask = F & E & S

    rows = {
        "Extreme\noverlap": intervals(overlap_mask),
        "Flood\nburden": intervals(dfp[flood_col]),
        "Erosion\nburden": intervals(dfp[erosion_col]),
        "Subsidence\nburden": intervals(dfp[subs_col]),
    }

    y_positions = {
        'Extreme\noverlap':21,
        "Flood\nburden": 14,
        "Erosion\nburden": 7,
        "Subsidence\nburden": 0,
    }

    colors = {
        'Extreme\noverlap': 'red',
        "Flood\nburden": "steelblue",
        "Erosion\nburden": "orange",
        "Subsidence\nburden": "#493e01",
    }

    for row_name, spans in rows.items():
        ax.broken_barh(
            spans,
            (y_positions[row_name], 7),
            facecolors=colors[row_name],
            linewidth=0
        )


    ax.set_yticks([y_positions[k] + 3.5 for k in rows])
    ax.set_yticklabels(rows.keys(),fontsize=12)

    for ylabel in ax.get_yticklabels():
        ylabel.set_horizontalalignment('center')

    ax.tick_params(axis='y', pad=40)

    ax.set_xlabel("Valid transect index",fontsize=14)
    ax.set_ylim(0, 28)
    ax.set_xlim(0, len(dfp))
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(label,fontsize=16)

    plt.tight_layout()
    plt.show()

    return fig

def sample_line(line, spacing):
    """Create equally spaced points along a LineString."""
    distances = np.arange(0, line.length + spacing, spacing)
    distances = distances[distances <= line.length]
    
    if distances[-1] < line.length:
        distances = np.append(distances, line.length)
    
    points = [line.interpolate(d) for d in distances]
    return distances, points


def extract_profiles_from_tif(
    tif_path,
    geojson_path,
    spacing=10,
    id_col=None,
):
    lines = gpd.read_file(geojson_path)

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs

        if lines.crs != raster_crs:
            lines = lines.to_crs(raster_crs)

        all_profiles = []

        for idx, row in lines.iterrows():
            geom = row.geometry

            if geom.geom_type == "MultiLineString":
                geoms = list(geom.geoms)
            else:
                geoms = [geom]

            for part_id, line in enumerate(geoms):
                dist, pts = sample_line(line, spacing)

                coords = [(p.x, p.y) for p in pts]
                values = [v[0] for v in src.sample(coords)]

                profile_id = row[id_col] if id_col else idx

                df = pd.DataFrame({
                    "profile_id": profile_id,
                    "part_id": part_id,
                    "distance_m": dist,
                    "x": [p.x for p in pts],
                    "y": [p.y for p in pts],
                    "value": values,
                })

                all_profiles.append(df)

    return pd.concat(all_profiles, ignore_index=True)

def profile_check(
    base_profiles,
    modified_profiles,
    profile_id=None,
    part_id=0,
    direction="left",
):
    # choose first profile if not specified
    if profile_id is None:
        profile_id = base_profiles["profile_id"].iloc[0]

    # select profile
    p0 = base_profiles[
        (base_profiles["profile_id"] == profile_id) &
        (base_profiles["part_id"] == part_id)
    ].copy()

    p1 = modified_profiles[
        (modified_profiles["profile_id"] == profile_id) &
        (modified_profiles["part_id"] == part_id)
    ].copy()

    # merge by distance
    p = p0[["distance_m", "value"]].rename(columns={"value": "z0"}).merge(
        p1[["distance_m", "value"]].rename(columns={"value": "z1"}),
        on="distance_m",
        how="inner"
    )

    p["z0"] = p["z0"].replace(-9999, np.nan).astype(float)
    p["z1"] = p["z1"].replace(-9999, np.nan).astype(float)
    p["diff"] = p["z1"] - p["z0"]

    x = p["distance_m"].to_numpy()
    z0 = p["z0"].to_numpy()
    z1 = p["z1"].to_numpy()
    diff = p["diff"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    ax1.plot(x, z0, "k", lw=2, label="original profile")
    ax1.plot(x, z1, "r", lw=2, label="modified profile")
    ax1.axhline(0, color="gray", ls="--", alpha=0.5)

    # try:
    #     shore0 = _zero_crossing(x, z0)
    #     shore1 = _zero_crossing(x, z1)

    #     dy = shore1 - shore0
    #     if direction == "right":
    #         dy = -dy

    #     ax1.axvline(shore0, color="k", ls=":")
    #     ax1.axvline(shore1, color="r", ls=":")

    #     ax1.annotate(
    #         f"shoreline shift = {dy:.2f} m",
    #         xy=(shore1, 0),
    #         xytext=(shore1, np.nanmax([z0, z1]) * 0.8),
    #         arrowprops=dict(arrowstyle="->"),
    #         ha="center"
    #     )

    #     print(f"profile_id={profile_id}, shoreline shift = {dy:.2f} m")

    # except Exception as e:
    #     print(f"Could not find shoreline crossing: {e}")

    ax1.set_ylabel("Elevation (m)",fontsize=16)
    ax1.legend(loc='lower left',fontsize=16)
    ax1.grid(alpha=0.3)

    ax2.plot(x, diff, color="blue")
    ax2.axhline(0, color="k", lw=1)

    
    mask = np.isfinite(diff)
    # Fill positive values (above zero) in blue
    ax2.fill_between(
        x, 0, diff,
        where=(diff >= 0) & mask,
        color="steelblue",
        alpha=0.5,
        interpolate=True
    )

    # Fill negative values (below zero) in red
    ax2.fill_between(
        x, 0, diff,
        where=(diff < 0) & mask,
        color="tomato",
        alpha=0.5,
        interpolate=True
    )


    if mask.any():
        ax2.set_xlim(x[mask].min(), x[mask].max())

    ax2.set_ylabel("Δz (m)",fontsize=16)
    ax2.set_xlabel("Distance along profile (m)",fontsize=16)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return p


def _zero_crossing(x, z):
    """Return first x-location where profile crosses z=0."""
    valid = np.isfinite(z)
    x = x[valid]
    z = z[valid]

    sign_change = np.where(np.diff(np.sign(z)) != 0)[0]

    if len(sign_change) == 0:
        raise ValueError("No zero crossing found")

    i = sign_change[0]

    return np.interp(
        0,
        [z[i], z[i + 1]],
        [x[i], x[i + 1]]
    )