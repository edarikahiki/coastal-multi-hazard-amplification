import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from pathlib import Path


def hazard_scoring(
    df,
    flood_column="flood_mean",
    erosion_column="sds:change_rate",
    subsidence_column="subsidence",
    bins=[0, 2, 7, 11, 15]
):
    """
    Calculate individual and combined coastal hazard scores for flooding,
    shoreline erosion, and land subsidence.

    Each hazard is classified into predefined severity classes and assigned
    a numerical score. The individual hazard scores are then summed to obtain
    a total hazard score (S), which is subsequently classified into low,
    moderate, high, and very high hazard categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the coastal hazard variables.

    flood_column : str, default="flood_mean"
        Column containing flood depth values in metres.

    erosion_column : str, default="sds:change_rate"
        Column containing shoreline change rates in metres per year.
        Negative values represent shoreline erosion, while positive values
        represent shoreline accretion.

    subsidence_column : str, default="subsidence"
        Column containing land subsidence rates.

    bins : list, default=[0, 2, 7, 11, 15]
        Boundaries used to classify the combined hazard score into low,
        moderate, high, and very high categories.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with the following additional columns:

        erosion :
            Absolute magnitude of negative shoreline change rates.

        S_E :
            Shoreline erosion severity score (0–5).

        S_S :
            Land subsidence severity score (1, 3, or 5).

        S_F :
            Coastal flooding severity score (1–5).

        S :
            Combined hazard score calculated as the sum of S_E, S_S,
            and S_F while ignoring missing values.

        category :
            Classification of the combined hazard score as low,
            moderate, high, or very high.

    Notes
    -----
    Missing values for individual hazards are assigned NaN and excluded from
    the combined score using ``numpy.nansum``. Consequently, the combined
    score can be calculated even when one or more hazard variables are
    unavailable.
    """

    # Allow either "column_name" or ["column_name"]
    if isinstance(flood_column, (list, tuple)):
        flood_column = flood_column[0]

    if isinstance(erosion_column, (list, tuple)):
        erosion_column = erosion_column[0]

    if isinstance(subsidence_column, (list, tuple)):
        subsidence_column = subsidence_column[0]

    # ------------------------------------------------------------------
    # Erosion
    # ------------------------------------------------------------------
    df["erosion"] = np.abs(df[erosion_column].clip(upper=0))

    df["S_E"] = np.select(
        [
            df[erosion_column].isna(),
            df[erosion_column] > 0,
            (df[erosion_column] >= -0.5) & (df[erosion_column] <= 0),
            (df[erosion_column] >= -1) & (df[erosion_column] < -0.5),
            (df[erosion_column] >= -3) & (df[erosion_column] < -1),
            (df[erosion_column] >= -5) & (df[erosion_column] < -3),
            df[erosion_column] < -5,
        ],
        [np.nan, 0, 1, 2, 3, 4, 5],
        default=np.nan
    )

    # ------------------------------------------------------------------
    # Subsidence
    # ------------------------------------------------------------------
    df["S_S"] = np.select(
        [
            df[subsidence_column].isna(),
            df[subsidence_column] <= 1,
            (df[subsidence_column] > 1) & (df[subsidence_column] <= 5),
            df[subsidence_column] > 5,
        ],
        [np.nan, 1, 3, 5],
        default=np.nan
    )

    # ------------------------------------------------------------------
    # Flooding
    # ------------------------------------------------------------------
    df["S_F"] = np.select(
        [
            df[flood_column].isna(),
            df[flood_column] < 0.5,
            (df[flood_column] >= 0.5) & (df[flood_column] < 1),
            (df[flood_column] >= 1) & (df[flood_column] < 2),
            (df[flood_column] >= 2) & (df[flood_column] <= 5),
            df[flood_column] > 5,
        ],
        [np.nan, 1, 2, 3, 4, 5],
        default=np.nan
    )

    # ------------------------------------------------------------------
    # Combined hazard score
    # ------------------------------------------------------------------
    df["S"] = np.nansum(
        df[["S_E", "S_S", "S_F"]],
        axis=1
    )

    labels = [
        "low (0–2)",
        "moderate (3–7)",
        "high (8–11)",
        "very high (12–15)"
    ]

    df["category"] = pd.cut(
        df["S"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

def assign_aggregation_framework(
    df,
    framework="h3",
    resolution=2,
    polygon_file=None,
):
    """
    Assign spatial aggregation units to a GeoDataFrame using either H3 cells
    or a user-defined polygon dataset.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input GeoDataFrame containing spatial features.

    framework : {"h3", "polygon"}, default="h3"
        Spatial aggregation framework to use.

        - ``"h3"`` assigns each feature centroid to an H3 cell.
        - ``"polygon"`` assigns features to polygons using a spatial join 
                (e.g. IPCC reference or country polygon).

    resolution : int, default=2
        H3 resolution used when ``framework="h3"``.

    polygon_file : str or pathlib.Path, optional
        Path to the polygon dataset used when ``framework="polygon"``.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the assigned spatial aggregation unit.

        For H3 aggregation, the output contains:
        - ``lon`` : centroid longitude
        - ``lat`` : centroid latitude
        - ``h3`` : H3 cell identifier

        For polygon aggregation, the polygon attributes are added through
        a spatial join.

    Raises
    ------
    ValueError
        If an unsupported framework is selected, ``polygon_file`` is not
        provided for polygon aggregation, or the polygon identifier column
        is missing.

    FileNotFoundError
        If the specified polygon file does not exist.
    """

    if not isinstance(df, gpd.GeoDataFrame):
        raise TypeError("df must be a GeoDataFrame.")

    if df.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS.")

    # ------------------------------------------------------------------
    # H3 aggregation
    # ------------------------------------------------------------------
    if framework == "h3":

        # Calculate centroids in a projected CRS to avoid geographic
        # centroid warnings / distortion
        centroid = df.to_crs(epsg=3857).geometry.centroid

        centroid = gpd.GeoSeries(
            centroid,
            crs="EPSG:3857"
        ).to_crs(epsg=4326)

        out_df = df.copy()

        out_df["lon"] = centroid.x.to_numpy()
        out_df["lat"] = centroid.y.to_numpy()

        # Assign centroid to H3 cell
        out_df["h3"] = [
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in zip(out_df["lat"], out_df["lon"])
        ]

    # ------------------------------------------------------------------
    # Polygon aggregation
    # ------------------------------------------------------------------
    elif framework == "polygon":

        if polygon_file is None:
            raise ValueError(
                "polygon_file must be provided when framework='polygon'."
            )

        polygon_path = Path(polygon_file)

        if not polygon_path.exists():
            raise FileNotFoundError(
                f"Polygon file not found: {polygon_path}"
            )

        # Read polygon dataset
        polygon = gpd.read_file(polygon_path)

        if polygon.crs is None:
            raise ValueError(
                "Polygon dataset must have a defined CRS."
            )

        # --------------------------------------------------------------
        # Filter unwanted polygons
        # --------------------------------------------------------------
        if "Type" in polygon.columns:
            polygon = polygon.loc[
                polygon["Type"] != "Ocean"
            ].copy()

        # --------------------------------------------------------------
        # Standardize CRS
        # --------------------------------------------------------------
        polygon = polygon.to_crs(4326)

        input_df = df.to_crs(4326).copy()

        # --------------------------------------------------------------
        # Calculate transect midpoint
        # --------------------------------------------------------------
        projected = df.to_crs(3857)

        midpoints = projected.geometry.interpolate(
            0.5,
            normalized=True
        )

        midpoints = gpd.GeoSeries(
            midpoints,
            index=df.index,
            crs=projected.crs
        ).to_crs(4326)

        # --------------------------------------------------------------
        # Use midpoint only for spatial assignment
        # --------------------------------------------------------------
        points = gpd.GeoDataFrame(
            input_df.drop(columns="geometry"),
            geometry=midpoints,
            crs="EPSG:4326"
        )

        # Polygon attributes to attach
        polygon_columns = [
            col for col in polygon.columns
            if col != "geometry"
        ]

        # Spatial join:
        # transect midpoint -> IPCC polygon attributes
        joined = gpd.sjoin(
            points,
            polygon[polygon_columns + ["geometry"]],
            how="left",
            predicate="within"
        )

        # --------------------------------------------------------------
        # Restore original transect geometry
        # --------------------------------------------------------------
        joined = joined.drop(
            columns=["geometry", "index_right"],
            errors="ignore"
        )

        joined["geometry"] = input_df.geometry

        out_df = gpd.GeoDataFrame(
            joined,
            geometry="geometry",
            crs="EPSG:4326"
        )

    # ------------------------------------------------------------------
    # Invalid framework
    # ------------------------------------------------------------------
    else:
        raise ValueError(
            "Incorrect framework. Select either 'h3' or 'polygon'."
        )

    return out_df




def calculate_burden(
    df,
    framework="h3",
    polygon_file=None,
    group_column="Acronym",
    min_transects=100
):
    """
    Calculate multi-hazard burden for spatial aggregation units.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input transect dataset containing hazard categories and spatial
        aggregation identifiers.

    framework : {"h3", "polygon"}, default="h3"
        Spatial aggregation framework.

    polygon_file : str or pathlib.Path, optional
        Path to the polygon dataset used when ``framework="polygon"``.

    group_column : str, default="Acronym"
        Column identifying polygon aggregation units. Only used when
        framework="polygon".

    min_transects : int, default=100
        Minimum number of valid transects required for an aggregation unit
        to be retained.

    Returns
    -------
    geopandas.GeoDataFrame
        Aggregated multi-hazard burden dataset containing hazard-category
        counts, relative concentrations, absolute burden, and MHB score.
    """

    # --------------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------------
    if "category" not in df.columns:
        raise ValueError(
            "Column 'category' is required. "
            "Run hazard_scoring() before calculate_burden()."
        )

    # --------------------------------------------------------------
    # H3 aggregation
    # --------------------------------------------------------------
    if framework == "h3":

        if "h3" not in df.columns:
            raise ValueError(
                "Column 'h3' not found. "
                "Run assign_aggregation_framework(..., framework='h3') first."
            )

        burden = (
            df.groupby("h3")
            .agg(
                n_low=(
                    "category",
                    lambda x: (x == "low (0–2)").sum()
                ),
                n_moderate=(
                    "category",
                    lambda x: (x == "moderate (3–7)").sum()
                ),
                n_high=(
                    "category",
                    lambda x: (x == "high (8–11)").sum()
                ),
                n_very_high=(
                    "category",
                    lambda x: (x == "very high (12–15)").sum()
                ),
            )
            .reset_index()
        )

        # H3 cell centre
        burden["lat"] = burden["h3"].apply(
            lambda x: h3.cell_to_latlng(x)[0]
        )

        burden["lon"] = burden["h3"].apply(
            lambda x: h3.cell_to_latlng(x)[1]
        )

        # Create point geometry at H3 cell centre
        burden_gdf = gpd.GeoDataFrame(
            burden,
            geometry=gpd.points_from_xy(
                burden["lon"],
                burden["lat"]
            ),
            crs="EPSG:4326"
        )

    # --------------------------------------------------------------
    # Polygon aggregation
    # --------------------------------------------------------------
    elif framework == "polygon":

        if polygon_file is None:
            raise ValueError(
                "polygon_file must be provided when framework='polygon'."
            )

        polygon_path = Path(polygon_file)
        

        if not polygon_path.exists():
            raise FileNotFoundError(
                f"Polygon file not found: {polygon_path}"
            )

        # Read polygon dataset
        polygon = gpd.read_file(polygon_path)

        if polygon.crs is None:
            raise ValueError(
                "Polygon dataset must have a defined CRS."
            )

        polygon = polygon.to_crs(4326)



        if group_column not in df.columns:
            raise ValueError(
                f"Grouping column '{group_column}' not found in dataframe."
            )
        
        burden = (
            df.groupby(group_column)
            .agg(
                n_low=("category", lambda x: (x == "low (0–2)").sum()),
                n_moderate=("category", lambda x: (x == "moderate (3–7)").sum()),
                n_high=("category", lambda x: (x == "high (8–11)").sum()),
                n_very_high=("category", lambda x: (x == "very high (12–15)").sum()),
            )
            .reset_index()
        )

        polygon_lookup = (
            polygon[
                [group_column, "geometry"]
            ]
            .drop_duplicates(subset=group_column)
        )

        burden_gdf = polygon_lookup.merge(
            burden,
            on=group_column,
            how="inner"
        )

        burden_gdf = gpd.GeoDataFrame(
            burden_gdf,
            geometry="geometry",
            crs=polygon.crs
        )

        # Standardise output CRS
        burden_gdf = burden_gdf.to_crs(4326)

    else:
        raise ValueError(
            "Incorrect framework. Select either 'h3' or 'polygon'."
        )

    # --------------------------------------------------------------
    # Common burden calculations
    # --------------------------------------------------------------

    # Total number of classified transects
    burden_gdf["n_total"] = (
        burden_gdf["n_low"]
        + burden_gdf["n_moderate"]
        + burden_gdf["n_high"]
        + burden_gdf["n_very_high"]
    )

    # Remove aggregation units with insufficient data
    burden_gdf = burden_gdf[
        burden_gdf["n_total"] > min_transects
    ].copy()

    # Percentage of high + very high hazard transects
    burden_gdf["p_high_vhigh"] = (
        burden_gdf["n_high"]
        + burden_gdf["n_very_high"]
    ) / burden_gdf["n_total"]

    # Percentage of very-high hazard transects
    burden_gdf["p_very_high"] = (
        burden_gdf["n_very_high"]
        / burden_gdf["n_total"]
    )

    # Relative concentration (%)
    burden_gdf["concentration"] = (
        burden_gdf["p_high_vhigh"] * 100
    )

    # Absolute number of high + very-high transects
    burden_gdf["n_abs"] = (
        burden_gdf["n_high"]
        + burden_gdf["n_very_high"]
    )

    # --------------------------------------------------------------
    # Multi-Hazard Burden (MHB)
    # --------------------------------------------------------------
    burden_gdf["MHB"] = (
        burden_gdf["concentration"]
        * np.log1p(burden_gdf["n_abs"])
    )

    # --------------------------------------------------------------
    # Clean output
    # --------------------------------------------------------------
    burden_gdf = burden_gdf.replace(
        [np.inf, -np.inf],
        np.nan
    )

    burden_gdf = burden_gdf.dropna(
        subset=["concentration", "n_abs", "MHB"]
    )

    burden_gdf = burden_gdf[
        burden_gdf["n_abs"] > 0
    ].copy()

    burden_gdf = burden_gdf[
        burden_gdf["MHB"] > 0
    ].copy()

    return burden_gdf


