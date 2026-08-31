# Coastal Multi-Hazard Amplification

This repository contains the analysis and modelling workflow developed
for an MSc thesis on **coastal multi-hazard screening and flood
amplification**. The framework combines globally available coastal
datasets describing **coastal flooding, shoreline change/erosion, and
land subsidence**, maps them to a common coastal transect system, and
evaluates their spatial co-occurrence and statistical association.

The workflow is intended as a **screening framework** based on harmonization of several hazard datasets. Global analysis is used
to identify locations where multiple severe coastal hazards coincide and
where extreme flooding is statistically amplified under concurrent
erosion and subsidence conditions. Selected locations can subsequently
be investigated with local hydrodynamic modelling.

Repository:
https://github.com/edarikahiki/coastal-multi-hazard-amplification

------------------------------------------------------------------------

## Workflow

The main analysis is organized as a sequence of Jupyter notebooks:

1.  **`01_hazard_mapping.ipynb`**\
    Maps the individual hazard datasets to the Global Coastal Transect
    System (GCTS). Raster-based flooding and subsidence information is
    sampled to the coastal transects and combined with shoreline-change
    information.

2.  **`02_mapped_hazard_visualization.ipynb`**\
    Visualizes the spatial and statistical distributions of the mapped
    flood, erosion, and subsidence datasets. Generalized coastline
    segments are used only for efficient global visualization; the
    analytical calculations retain the transect-level information.

3.  **`03_statistical_analysis_documented.ipynb`**\
    Performs the main multi-hazard screening analysis. The workflow
    assigns transects to a common spatial aggregation framework,
    identifies locally extreme hazard conditions, calculates
    Multi-Hazard Burden and statistical flood amplification, and
    combines both components into the Hotspot Priority Index (HPI).

4.  **`04_scaling_example.ipynb`**\
    Demonstrates the spatial scalability of the framework. The analysis
    can use alternative polygon-based aggregation regions instead of H3
    cells and can also be applied to a smaller domain using a finer H3
    resolution.

The modelling part of the repository uses **HydroMT-SFINCS / SFINCS** to
construct terrain scenarios representing erosion, subsidence, and their
combination and to evaluate the resulting changes in coastal flooding.

------------------------------------------------------------------------

## Main indicators

### Multi-Hazard Burden

The Multi-Hazard Burden (`MHB`) describes the concentration of severe
hazard conditions within a spatial aggregation unit. Flooding, erosion,
and subsidence are first classified relative to their local hazard
distributions, after which their combined occurrence is summarized at
the regional scale.

### Statistical amplification

Flood amplification is evaluated from the change in the probability of
extreme flooding when other extreme hazards are present. The compound
erosion-subsidence amplification factor is conceptually expressed as

$$
A_{F|ES} = \frac{P(F \mid E \cap S)}{P(F)}
$$

where (F), (E), and (S) represent extreme flooding, erosion, and
subsidence, respectively. Values greater than one indicate that extreme
flooding occurs more frequently under the specified compound hazard
condition than in the background population.

### Hotspot Priority Index

The Hotspot Priority Index (`HPI`) combines normalized Multi-Hazard
Burden and amplification information to identify locations where severe
multi-hazard conditions and statistical amplification occur together. It
is intended as a relative screening and prioritization indicator.

------------------------------------------------------------------------

## Python environment

Create the Conda environment supplied with the repository, if available:

``` bash
conda env create -f environment.yml
conda activate multi-hazard
```

The analysis relies primarily on the scientific Python and geospatial
ecosystem, including packages such as:

-   `numpy`
-   `pandas`
-   `geopandas`
-   `xarray`
-   `rasterio`
-   `rioxarray`
-   `shapely`
-   `pyproj`
-   `h3`
-   `scipy`
-   `matplotlib`
-   `cartopy`
-   `pyarrow`
-   `hydromt`
-   `hydromt-sfincs`

Package versions should preferably follow the environment file included
with the repository.

------------------------------------------------------------------------

# Data

The input datasets are **not all distributed with this GitHub
repository** because several are large external datasets or model
inputs. Download the original datasets from their providers and
reproduce the folder structure expected by the notebooks.

A typical local data directory used by the workflow is:

``` text
data/
├── base_model/
├── coastlines_osm_generalized_v2023/
├── Floodmap_noveg_rp100/
├── GSWO/
├── IPCC region/
├── model_input/
├── SFINCS_v2.3.0_mt_Faber_release_exe/
├── Subsidence/
├── transformed_terrain/
├── data_catalog.yml
├── mapped_hazard.parquet
├── mapped_hazard_backup.parquet
├── model_terrain.yml
└── run.bat
```

The directory names above correspond to the local organization used
during the thesis. Some folders contain **original external data**,
while others are **derived products generated by the notebooks or
modelling workflow**.

## Data sources and folder contents

 ## Data sources and folder contents

| Folder / file | Purpose | Source / how to obtain |
|---|---|---|
| `coastlines_osm_generalized_v2023/` | Generalized coastline used for efficient global visualization and spatial operations. | Derived from **OpenStreetMap coastline data**. Global processed coastline lines can be downloaded from `https://osmdata.openstreetmap.de/data/coastlines.html`. OpenStreetMap-derived coastline data are distributed under the ODbL. The generalized 2023 version used in this project is a processed derivative and may need to be recreated from the appropriate historical/source coastline if exact reproduction is required. |
| `Floodmap_noveg_rp100/` | Global coastal flood hazard used to map flood depth to GCTS transects. The project uses the no-vegetation, 100-year return-period flood layer. | Based on the global coastal flooding dataset/methodology of **van Zelst et al. (2026), *Adding a new dimension to the global flood protection value of mangroves and tidal marshes***. The exact prepared raster used by this repository is not redistributed here; obtain the source flood product from the original data provider, project archive, or authors where required. |
| `GSWO/` | Global Surface Water Occurrence data used during preparation of the local SFINCS terrain/model domain and land-water masking. | **JRC Global Surface Water** dataset by Pekel et al. Download from `https://global-surface-water.appspot.com/download`. Dataset information is available at `https://data.jrc.ec.europa.eu/dataset/jrc-gswe-global-surface-water-explorer-v1`. |
| `IPCC region/` | Polygon regions used to demonstrate that the multi-hazard framework can operate with an externally defined polygon aggregation framework instead of H3. | **IPCC AR6 WGI Reference Regions v4**. The GeoJSON and shapefile are available from the IPCC-WG1 Atlas repository at `https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions`. |
| `Subsidence/` | Global land-subsidence dataset mapped to the coastal transects. | **Hasan et al. (2023), *Global land subsidence mapping reveals widespread loss of aquifer storage capacity***, *Nature Communications*, 14, 6180. Dataset archive: `https://doi.org/10.4211/hs.dc7c5bfb3a86479b889d3b30ab0e4ef7`. Project code: `https://github.com/mdfahimhasan/Global-Subsidence-Groundwater`. |
| `SFINCS_v2.3.0_mt_Faber_release_exe/` | SFINCS executable used for the local flood simulations. | **Deltares SFINCS v2.3.0 mt Faber release**. Releases are available at `https://github.com/Deltares/SFINCS/releases`. The precompiled executable is available from `https://download.deltares.nl/en/sfincs/`. Use the release consistent with the model configuration when reproducing the simulations. |
| `base_model/` | Baseline SFINCS model(s) from which the terrain-change scenarios are constructed. | **Baseline flood model scenario.** Build using the modelling workflow and the external elevation, bathymetry, shoreline, surface-water, and forcing datasets specified by the HydroMT data catalog. This folder is based on **van Zelst et al. (2026)** and is not an independently downloadable external dataset. |
| `model_input/` | Prepared input files used to construct or run the local SFINCS models. | **Generated / assembled project data.** Contents depend on the selected case-study domain and HydroMT-SFINCS configuration. |
| `transformed_terrain/` | Terrain rasters generated for erosion, subsidence, and combined future-terrain scenarios. | **Generated by this repository.** These files are outputs of the terrain-transformation workflow and should be recreated from the baseline terrain and mapped hazard information. |
| `mapped_hazard.parquet` | Integrated GCTS-level dataset containing the mapped flood, erosion/shoreline-change, and subsidence variables used by the statistical analysis. | **Generated by `01_hazard_mapping.ipynb`.** It is not a primary external dataset. |
| `data_catalog.yml` | HydroMT data catalog describing model input datasets. | Project configuration file. Paths may need to be adapted to the local data directory. |
| `model_terrain.yml` | HydroMT catalog/configuration for generated terrain scenarios. | Project configuration file generated or updated by the modelling workflow. |
| `run.bat` | Convenience script for running SFINCS on Windows. | Project utility file; adjust the executable and model paths if necessary. |

## Global Coastal Transect System and shoreline-change data

The common spatial support for the global analysis is the **Global
Coastal Transect System (GCTS)** and associated Global Coastal Transect
Repository (GCTR). The repository provides analysis-ready coastal
transects and coastal attributes, including the shoreline-change
information used in this workflow.

The GCTR can be obtained from:

-   Zenodo: https://zenodo.org/records/16928269
-   CoastPy project: https://github.com/TUDelft-CITG/coastpy

When using the GCTS/GCTR, cite:

> Calkoen, F. R., Luijendijk, A. P., Vos, K., Kras, E., & Baart, F.
> (2025). Enabling coastal analytics at planetary scale. *Environmental
> Modelling & Software*, 183, 106257.
> https://doi.org/10.1016/j.envsoft.2024.106257

The notebooks may access the transect repository separately rather than
storing the complete global dataset inside the local `data/` directory.
To access data contained in GCTR, an access code is needed and can be 
obtained by contacting Deltares.

------------------------------------------------------------------------

## Recommended data preparation order

For a clean reproduction, prepare the datasets in approximately the
following order:

1.  Obtain the **GCTS/GCTR** transects and shoreline-change attributes.
2.  Download the global **subsidence** dataset.
3.  Obtain the **100-year coastal flood** dataset used by the analysis.
4.  Download the **OSM coastline** data if the global visualization
    notebooks are to be reproduced.
5.  Run `01_hazard_mapping.ipynb` to create `mapped_hazard.parquet`.
6.  Download the **IPCC reference regions** for the polygon-scaling
    example.
7.  For local SFINCS modelling, obtain/generate SFINCS baseline model and the
    required **GSWO** tiles, then prepare the HydroMT data catalog.
8.  Build the `base_model/`.
9.  Generate the erosion/subsidence terrain scenarios under
    `transformed_terrain/`.
10. Run the corresponding SFINCS scenarios and evaluate the resulting
    flood maps.

------------------------------------------------------------------------

## Notes on reproducibility

-   The global datasets have different spatial coverage and data
    availability. Missing values should therefore not automatically be
    interpreted as absence of a hazard.
-   The generalized OSM coastline is used for **visualization**, not as
    the analytical aggregation framework for the global multi-hazard
    calculations.
-   `mapped_hazard.parquet`, `model_input/`, and
    `transformed_terrain/` are derived project products and can be
    regenerated when the required source datasets are available.
-   Relative paths are recommended in the HydroMT YAML catalogs where
    possible. If HydroMT resolves paths relative to the working
    directory in a particular setup, update the catalog `root` before
    running the model.
-   Large external datasets and SFINCS executables should generally
    remain outside Git version control.

------------------------------------------------------------------------

## Key references

-   Calkoen, F. R., Luijendijk, A. P., Vos, K., Kras, E., & Baart, F.
    (2025). **Enabling coastal analytics at planetary scale.**
    *Environmental Modelling & Software*, 183, 106257.
    https://doi.org/10.1016/j.envsoft.2024.106257
-   Hasan, M. F., Smith, R., Vajedian, S., Pommerenke, R., &
    Majumdar, S. (2023). **Global land subsidence mapping reveals
    widespread loss of aquifer storage capacity.** *Nature
    Communications*, 14, 6180.
    https://doi.org/10.1038/s41467-023-41933-z
-   Kirezci, E., Young, I. R., Ranasinghe, R., Muis, S., Nicholls, R.
    J., Lincke, D., & Hinkel, J. (2020). **Projections of global-scale
    extreme sea levels and resulting episodic coastal flooding over the
    21st Century.** *Scientific Reports*, 10, 11629.
    https://doi.org/10.1038/s41598-020-67736-6
-   Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016).
    **High-resolution mapping of global surface water and its long-term
    changes.** *Nature*, 540, 418--422.
    https://doi.org/10.1038/nature20584
-   IPCC-WG1 Atlas. **AR6 WGI Reference Regions v4.**
    https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions
-   Deltares. **SFINCS.** https://github.com/Deltares/SFINCS

------------------------------------------------------------------------

## Citation

If this repository is used in research, please cite the MSc thesis
associated with the repository together with the original datasets and
software listed above. Dataset-specific licenses and citation
requirements remain applicable to all externally sourced data.

## License

See the repository license for the code in this project. External
datasets and software retain their own licenses and terms of use.
