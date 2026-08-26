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


def classify_extreme(
    df,
    flood_column="flood_mean",
    erosion_column="sds:change_rate",
    subsidence_column="subsidence",
    quantile=0.9,
):
    """
    Classify extreme flood, erosion, and subsidence values using
    quantile-based thresholds.

    Shoreline erosion is represented as the absolute value of negative
    shoreline change rates, such that positive or stable shoreline
    changes are assigned an erosion magnitude of zero. Extreme conditions
    are defined independently for each hazard using the specified
    quantile of the available values.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Input dataframe containing flood depth, shoreline change rate,
        and land subsidence data.
    flood_column : str, default "flood_mean"
        Column containing flood values.
    erosion_column : str, default "sds:change_rate"
        Column containing shoreline change rates. Negative values are
        interpreted as erosion.
    subsidence_column : str, default "subsidence"
        Column containing land subsidence values.
    quantile : float, default 0.9
        Quantile used to define extreme hazard conditions. For example,
        0.9 defines values at or above the 90th percentile as extreme.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Copy of the input dataframe with the following additional columns:

        - ``erosion`` : absolute magnitude of negative shoreline change.
        - ``flood`` : flood values copied from ``flood_column``.
        - ``flood_extreme`` : boolean indicator of extreme flooding.
        - ``erosion_extreme`` : boolean indicator of extreme erosion.
        - ``subs_extreme`` : boolean indicator of extreme subsidence.

    Notes
    -----
    Thresholds are calculated independently from the distribution of
    each hazard variable in the supplied dataframe. Therefore, if the
    dataframe contains only a regional subset, the resulting thresholds
    represent regional rather than global extreme conditions.
    """
    df = df.copy()

    # Convert negative shoreline change rates to positive erosion magnitude
    df["erosion"] = df[erosion_column].clip(upper=0).abs()
    df["flood"] = df[flood_column]
    df["subsidence"] = df[subsidence_column]

    # Calculate quantile thresholds
    flood_threshold = df["flood"].quantile(quantile)
    erosion_threshold = df["erosion"].quantile(quantile)
    subsidence_threshold = df[subsidence_column].quantile(quantile)

    # Classify extreme hazard conditions
    df["flood_extreme"] = df["flood"] >= flood_threshold
    df["erosion_extreme"] = df["erosion"] >= erosion_threshold
    df["subs_extreme"] = df[subsidence_column] >= subsidence_threshold

    print(
        "Flood threshold:", flood_threshold,
        "Erosion threshold:", erosion_threshold,
        "Subsidence threshold:", subsidence_threshold,
    )

    return df

def amp_stats(
    df,
    min_total=100,
    min_overlap=10,
):
    """
    Calculate flood amplification statistics for a single spatial group.

    Amplification is defined as:

        A_F|E  = P(F | E) / P(F)
        A_F|S  = P(F | S) / P(F)
        A_F|ES = P(F | E ∩ S) / P(F)

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Data for a single spatial aggregation unit. Missing-value
        handling should be performed before calling this function.

    min_total : int, default 100
        Minimum number of observations required to calculate
        amplification.

    min_overlap : int, default 10
        Minimum number of observations in the corresponding conditioning
        population.

    Returns
    -------
    pandas.Series
        Observation counts, conditional probabilities, and amplification
        factors for the spatial group.
    """

    # Extreme hazard indicators
    F = df["flood_extreme"].fillna(False).astype(bool)
    E = df["erosion_extreme"].fillna(False).astype(bool)
    S = df["subs_extreme"].fillna(False).astype(bool)

    # Hazard combinations
    FE = F & E
    FS = F & S
    ES = E & S
    FES = F & E & S

    # Counts
    n_total = len(df)

    n_F = F.sum()
    n_E = E.sum()
    n_S = S.sum()

    n_FE = FE.sum()
    n_FS = FS.sum()
    n_ES = ES.sum()
    n_FES = FES.sum()

    # Baseline flood probability
    pF = F.mean() if n_total > 0 else np.nan

    # Default values
    pF_E = np.nan
    pF_S = np.nan
    pF_ES = np.nan

    amp_FE = np.nan
    amp_FS = np.nan
    amp_FES = np.nan

    # Calculate amplification only when sufficient data are available
    if (
        n_total >= min_total
        and pd.notna(pF)
        and pF > 0
    ):

        # Flood given erosion
        if n_E >= min_overlap:
            pF_E = F[E].mean()
            amp_FE = pF_E / pF

        # Flood given subsidence
        if n_S >= min_overlap:
            pF_S = F[S].mean()
            amp_FS = pF_S / pF

        # Flood given erosion and subsidence
        if n_ES >= min_overlap:
            pF_ES = F[ES].mean()
            amp_FES = pF_ES / pF

    return pd.Series({
        "n_total": n_total,

        "n_F": n_F,
        "n_E": n_E,
        "n_S": n_S,

        "n_FE": n_FE,
        "n_FS": n_FS,
        "n_ES": n_ES,
        "n_FES": n_FES,

        "pF": pF,
        "pF_E": pF_E,
        "pF_S": pF_S,
        "pF_ES": pF_ES,

        "amp_FE": amp_FE,
        "amp_FS": amp_FS,
        "amp_FES": amp_FES,
    })

def calculate_amplification(
    df,
    framework="h3",
    polygon_file=None,
    group_column="Acronym",
    min_total=100,
    min_overlap=10,
    complete_case=True,
):
    """
    Calculate regional flood amplification using either H3 cells
    or user-supplied polygon aggregation.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Transect-level hazard dataset containing flood, erosion,
        subsidence, extreme-hazard classifications, and geometry.

    framework : {"h3", "polygon"}, default "h3"
        Spatial aggregation framework.

        - ``"h3"`` groups observations using the existing H3 identifier.
        - ``"polygon"`` spatially joins observations to an external
          polygon dataset and groups them using ``group_column``.

    polygon_file : str or pathlib.Path, optional
        Path to the polygon dataset. Required when
        ``framework="polygon"``.

    group_column : str, default "Acronym"
        Polygon identifier used for grouping when
        ``framework="polygon"``.

    min_total : int, default 100
        Minimum number of observations within each spatial unit
        required to calculate amplification.

    min_overlap : int, default 10
        Minimum number of conditioning observations required to
        calculate each amplification factor.

    complete_case : bool, default True
        If True, observations missing flood, erosion, or subsidence
        values are removed before spatial aggregation and amplification
        calculation.

    Returns
    -------
    geopandas.GeoDataFrame
        Spatially aggregated amplification statistics with geometry.
    """

    df = df.copy()

    # --------------------------------------------------------------
    # Handle missing observations
    # --------------------------------------------------------------
    if complete_case:
        df = df.dropna(
            subset=["flood", "erosion", "subsidence"]
        ).copy()

    # --------------------------------------------------------------
    # H3 aggregation
    # --------------------------------------------------------------
    if framework == "h3":

        if "h3" not in df.columns:
            raise KeyError(
                "'h3' column is required when framework='h3'."
            )

        # Calculate statistics
        df_amp = (
            df.groupby("h3")
            .apply(
                amp_stats,
                min_total=min_total,
                min_overlap=min_overlap,
                include_groups=False,
            )
            .reset_index()
        )

        # H3 cell centre
        df_amp["lat"] = df_amp["h3"].apply(
            lambda x: h3.cell_to_latlng(x)[0]
        )

        df_amp["lon"] = df_amp["h3"].apply(
            lambda x: h3.cell_to_latlng(x)[1]
        )

        # Create point geometry at H3 cell centre
        df_amp = gpd.GeoDataFrame(
            df_amp,
            geometry=gpd.points_from_xy(
                df_amp["lon"],
                df_amp["lat"]
            ),
            crs="EPSG:4326"
        )

    # --------------------------------------------------------------
    # Polygon aggregation
    # --------------------------------------------------------------
    elif framework == "polygon":

        if polygon_file is None:
            raise ValueError(
                "polygon_file must be provided when "
                "framework='polygon'."
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

        if group_column not in polygon.columns:
            raise KeyError(
                f"'{group_column}' not found in polygon dataset."
            )

        # Remove ocean polygons if present
        if "Type" in polygon.columns:
            polygon = polygon[
                polygon["Type"] != "Ocean"
            ].copy()

        # Match CRS
        if df.crs != polygon.crs:
            polygon = polygon.to_crs(df.crs)

        # ----------------------------------------------------------
        # Spatially assign transects to polygons
        # ----------------------------------------------------------
        df_joined = gpd.sjoin(
            df,
            polygon[[group_column, "geometry"]],
            how="inner",
            predicate="intersects",
        )

        # Calculate statistics by polygon
        df_amp = (
            df_joined.groupby(group_column)
            .apply(
                amp_stats,
                min_total=min_total,
                min_overlap=min_overlap,
                include_groups=False,
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Attach polygon geometry
        # ----------------------------------------------------------
        polygon_geometry = (
            polygon[[group_column, "geometry"]]
            .dissolve(by=group_column)
            .reset_index()
        )

        df_amp = polygon_geometry.merge(
            df_amp,
            on=group_column,
            how="left",
        )

        df_amp = gpd.GeoDataFrame(
            df_amp,
            geometry="geometry",
            crs=polygon.crs,
        )

    else:
        raise ValueError(
            "framework must be either 'h3' or 'polygon'."
        )

    return df_amp


def calculate_HPI(
    burden,
    amplification,
    amp_column="amp_FES",
    framework="h3",
    polygon_id="Acronym",
    burden_column="MHB",
):
    """
    Calculate the Hotspot Prioritization Index (HPI) by combining
    Multi-Hazard Burden (MHB) and statistical flood amplification.

    The two constituent indicators are normalized independently using
    min-max normalization and subsequently combined multiplicatively:

        N_X = (X - X_min) / (X_max - X_min)

        HPI = N_MHB * N_amp

    The function supports both H3-based and polygon-based spatial
    aggregation.

    Parameters
    ----------
    burden : pandas.DataFrame or geopandas.GeoDataFrame
        Dataframe containing the Multi-Hazard Burden values.

        For ``framework="h3"``, it must contain an ``h3`` column.
        For ``framework="polygon"``, it must contain the column specified
        by ``polygon_id``.

    amplification : pandas.DataFrame or geopandas.GeoDataFrame
        Dataframe containing the statistical amplification results and
        the spatial identifier corresponding to ``burden``.

    amp_column : str, default "amp_FES"
        Amplification column used to construct the HPI. For example:

        - ``amp_FE``  : flood amplification conditioned on erosion
        - ``amp_FS``  : flood amplification conditioned on subsidence
        - ``amp_FES`` : flood amplification conditioned on simultaneous
          erosion and subsidence

    framework : {"h3", "polygon"}, default "h3"
        Spatial aggregation framework used for the calculation.

        - ``"h3"`` joins the data using the ``h3`` identifier.
        - ``"polygon"`` joins the data using ``polygon_id``.

    polygon_id : str, default "Acronym"
        Identifier column used to join polygon-based results.
        Ignored when ``framework="h3"``.

    burden_column : str, default "MHB"
        Column containing the Multi-Hazard Burden values.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Combined dataframe containing the burden, amplification,
        normalized indicators, and HPI:

        - ``amp``   : selected amplification factor
        - ``N_amp`` : normalized amplification
        - ``N_mhb`` : normalized Multi-Hazard Burden
        - ``HPI``   : Hotspot Prioritization Index

        If ``burden`` is a GeoDataFrame, its geometry is preserved.

    Raises
    ------
    ValueError
        If an unsupported framework is provided.

    KeyError
        If required identifier or metric columns are missing.
    """

    # --------------------------------------------------------------
    # Copy inputs to avoid modifying original dataframes
    # --------------------------------------------------------------
    burden = burden.copy()
    amplification = amplification.copy()

    # --------------------------------------------------------------
    # Determine spatial identifier
    # --------------------------------------------------------------
    if framework == "h3":
        join_column = "h3"

    elif framework == "polygon":
        join_column = polygon_id

    else:
        raise ValueError(
            "framework must be either 'h3' or 'polygon'."
        )

    # --------------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------------
    if join_column not in burden.columns:
        raise KeyError(
            f"'{join_column}' not found in burden dataframe."
        )

    if join_column not in amplification.columns:
        raise KeyError(
            f"'{join_column}' not found in amplification dataframe."
        )

    if burden_column not in burden.columns:
        raise KeyError(
            f"'{burden_column}' not found in burden dataframe."
        )

    if amp_column not in amplification.columns:
        raise KeyError(
            f"'{amp_column}' not found in amplification dataframe."
        )

    # --------------------------------------------------------------
    # Select amplification metric
    # --------------------------------------------------------------
    amplification["amp"] = amplification[amp_column]

    # Only retain columns needed from amplification.
    # This prevents duplicate columns such as geometry, counts, etc.
    amp_keep = [
        join_column,
        "amp",
    ]

    # Optionally retain useful amplification statistics
    optional_columns = [
        "n_total",
        "n_F",
        "n_E",
        "n_S",
        "n_FE",
        "n_FS",
        "n_ES",
        "n_FES",
        "pF",
        "pF_E",
        "pF_S",
        "pF_ES",
        "amp_FE",
        "amp_FS",
        "amp_FES",
    ]

    for col in optional_columns:
        if (
            col in amplification.columns
            and col not in amp_keep
        ):
            amp_keep.append(col)

    amplification = amplification[amp_keep]

    # --------------------------------------------------------------
    # Merge burden and amplification
    # --------------------------------------------------------------
    hpi = burden.merge(
        amplification,
        on=join_column,
        how="left",
    )

    # --------------------------------------------------------------
    # Min-max normalization helper
    # --------------------------------------------------------------
    def minmax(series):
        valid = series.dropna()

        if valid.empty:
            return pd.Series(
                np.nan,
                index=series.index,
                dtype=float,
            )

        xmin = valid.min()
        xmax = valid.max()

        # Avoid division by zero when all values are identical
        if xmax == xmin:
            return pd.Series(
                np.nan,
                index=series.index,
                dtype=float,
            )

        return (series - xmin) / (xmax - xmin)

    # --------------------------------------------------------------
    # Normalize constituent indicators
    # --------------------------------------------------------------
    hpi["N_amp"] = minmax(hpi["amp"])
    hpi["N_mhb"] = minmax(hpi[burden_column])

    # --------------------------------------------------------------
    # Calculate HPI
    # --------------------------------------------------------------
    hpi["HPI"] = (
        hpi["N_amp"]
        * hpi["N_mhb"]
    )

    # --------------------------------------------------------------
    # Optional global HPI ranking
    # --------------------------------------------------------------
    hpi["HPI_rank"] = hpi["HPI"].rank(
        method="min",
        ascending=False,
    )

    # Keep missing HPI as NaN but use nullable integer for rank
    hpi["HPI_rank"] = hpi["HPI_rank"].astype("Int64")

    return hpi