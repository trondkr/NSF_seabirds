import os
import os.path
import sys
import warnings

import cartopy.geodesic as cgd
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shapely
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import colors
from shapely import geometry

sys.path.insert(0, '../.')

warnings.filterwarnings('ignore')
plt.style.use('mpl20')

import xarray as xr
import numpy as np


def get_etopo1():
    etopo1_filename = "/home/jovyan/shared/etopo1/ETOPO1_Ice_g_gmt4.grd"
    ds_topo = xr.open_dataset(etopo1_filename)
    topo = ds_topo.z
    ds_topo.close()

    topo2 = topo.sel(y=slice(-90, 90), x=slice(-180, 180))
    x = topo2.x  # 21601
    y = topo2.y  # 10801
    X, Y = np.meshgrid(x, y)
    return X, Y, topo2


def level_colormap(levels, cmap=None):
    """Make a colormap based on an increasing sequence of levels"""

    # Start with an existing colormap
    if cmap == None:
        cmap = plt.get_cmap()

    # Spread the colours maximally
    nlev = len(levels)
    S = np.arange(nlev, dtype="float") / (nlev - 1)
    A = cmap(S)

    # Normalize the levels to interval [0,1]
    levels = np.array(levels, dtype="float")
    L = (levels - levels[0]) / (levels[-1] - levels[0])

    # Make the colour dictionary
    R = [(L[i], A[i, 0], A[i, 0]) for i in range(nlev)]
    G = [(L[i], A[i, 1], A[i, 1]) for i in range(nlev)]
    B = [(L[i], A[i, 2], A[i, 2]) for i in range(nlev)]
    cdict = dict(red=tuple(R), green=tuple(G), blue=tuple(B))

    # Use
    return colors.LinearSegmentedColormap("%s_levels" % cmap.name, cdict, 256)


def create_map(stations, dfs_all, var_name):
    projection = ccrs.PlateCarree(central_longitude=-180)
    ax = plt.figure(figsize=(6, 4)).gca(projection=projection)
    ax.coastlines(resolution="110m", linewidth=0.6, color="black", alpha=0.8, zorder=10)
    ax.add_feature(cpf.BORDERS, linestyle=":", alpha=0.4)
    ax.add_feature(cpf.LAND, color="lightgrey", zorder=9)
    ax.set_extent([-180, 180, -90, 90])

    #  xticks = np.linspace(-180, -120, 5)
    #  yticks = np.linspace(46, 62, 5)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    """
    X, Y, topo2 = get_etopo1()

    plt.grid(True, zorder=0, alpha=0.5)
    levels = [
        -8000,
        -6000,
        -4500,
        -3500,
        -3000,
        -2500,
        -2000,
        -1500,
        -1000,
        -500,
        -200,
        -175,
        -150,
        -125,
        -100,
        -75,
        -50,
        -40,
        -30,
        -20,
        -10,
    ]

    cm = level_colormap(levels, cmap=cmocean.cm.deep_r)
    cs3 = ax.contourf(
        X,
        Y,
        topo2,
        cmap=cm,
        levels=levels,
        zorder=3,
        alpha=0.8,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    """
    for station_key in stations.keys():
        station = stations[station_key]
        lat = station["lat"]
        lon = station["lon"]

        print("Scatter", station_key, lat, lon)
        ax.scatter(lon, lat, 50, zorder=30, c="red", transform=ccrs.PlateCarree())

        list_of_lons = stations["lons"]
        list_of_lats = stations["lats"]

        ax.scatter(list_of_lons, list_of_lats, 10, zorder=30, c="green", transform=ccrs.PlateCarree())

        plotfile = "Figures/NSF_map_stations.png"
    plt.savefig(plotfile, dpi=200)
    print("Saved map to {}".format(plotfile))


def create_map_of_sites(ds):
    lonmin = -180
    lonmax = 180
    latmin = -90
    latmax = 90

    palette = sns.color_palette("bright")
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-180))
    ax.coastlines(resolution="10m", linewidth=0.6, color="black", alpha=0.8, zorder=0)
    # ax.add_feature(cpf.BORDERS, linestyle=':',alpha=0.6)
    ax.add_feature(cpf.LAND, color="lightgrey")
    ax.set_extent([lonmin, lonmax, latmin, latmax])

    # plot the lat lon labels
    # https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
    # https://stackoverflow.com/questions/49956355/adding-gridlines-using-cartopy
    xticks = np.linspace(lonmin, lonmax, 5)
    yticks = np.linspace(latmin, latmax, 5)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    sns.set_palette(sns.color_palette("bright"))
    radius = 300000  # in m

    for station in stations.keys():
        st=stations[station]
        print(st)
        ax.scatter(x=st["lon"], y=st["lat"],
                    color="red",
                    s=10,
                    alpha=0.8,
                    transform=ccrs.PlateCarree())

        circle_points = cgd.Geodesic().circle(lon=float(st["lon"]),
                                                           lat=float(st["lat"]),
                                                           radius=radius,
                                                           n_samples=180, endpoint=False)
      #  geom = shapely.geometry.Polygon(circle_points)
       # ax.add_geometries((geom,), crs=ccrs.PlateCarree(), alpha=0.2, facecolor='red', edgecolor='black',
       #                   linewidth=0.01)

    if not os.path.exists("Figures"):
        os.makedirs("Figures", exist_ok=True)

    plotfile="Figures/Map_seabird_colonies_NSF.png"
    plt.savefig(plotfile, dpi=200)
    print(f"Finished creating map of seabird sites with radius {radius}m : {plotfile}")

### MAIN
infile = "site_coordinateswID_111622_21Nov2022.csv"
df = pd.read_csv(infile, header=[0])
print(df.head())
base_directory = "../../shared/NSF"
if not os.path.exists(base_directory):
    os.makedirs(base_directory, exist_ok=True)

stations = {}
maxindex = 90
# Prepare all the stations
for index, row in df.iterrows():

    st_lat = float(row.Latitude)
    st_lon = float(row.Longitude)

    station = {"station_name": row.Site,
               "lat": st_lat,
               "lon": st_lon,
               "ID": row.ID,
               "base_directory": base_directory}
    stations[row.Site] = station

    print(f"Station: {row.Site} latitude: {st_lat} longitude: {st_lon}")
    if index == maxindex:
        break


create_map_of_sites(stations)