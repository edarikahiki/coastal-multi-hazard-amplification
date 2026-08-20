import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

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