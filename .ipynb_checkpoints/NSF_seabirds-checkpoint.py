import os
import os.path
import sys
import warnings

import gcsfs
import global_land_mask
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xesmf as xe
from google.cloud import storage
from zarr.convenience import consolidate_metadata

sys.path.insert(0, '../.')

warnings.filterwarnings('ignore')
plt.style.use('mpl20')

import gsw_xarray as gsw
import xarray as xr
import numpy as np

"""Calculate stratification and properties of water surrounding seabird sites

Stratification is done using Brunt-Vaisala frequency as also done in 
"Increasing ocean stratification over the past  half-century"

*Notes from NSF review:*
Reviewers for the proposal suggests several aveanues towards making this project successful:
1. How about assessing just high stratification years vs low stratification years and not hang 
everything on climate change trends
2. From a mechanistic point of view, does the magnitude of stratification change result in a change in 
nutrient supply to the euphotic zone that could potentially account for changes in seabird productivity? 
This latter is a mass balance approach that is not proposed but could be assessed in a back-of-the-envelope 
fashion to give us some confidence that such a mechanistic link could in fact be responsible for the presumed linkages.
3. Is the 5 vs 200m stratification (Figure 2) the actual metric of stratification that will be used for analysis? If 
so, how will you deal with the sites (Figure 3) that are over the broad shelves that have 
bottom depths << 200 m? If not, how will you define this
4. The authors focus solely on thermal stratification, but at high latitudes – especially in the North Pacific where 
many of their key datasets originate – salinity is more often than not the primary determinant of stratification over 
the 5-200 m depth range. (In summer, thermal stratification may dominate over the 5-35 m euphotic zone depth layer.) 
The authors seek mechanistic links between climate change-induced stratification changes and the seabirds. 
The regulation of nutrients through the pycnocline via diapycnal mixing is never directly mentioned, but this is 
the most obvious and straightforward mechanistic link between stratification and productivity. Stratification in 
its entirety (both thermal and haline) is the regulator of nutrient fluxes so focusing on the thermal aspects only 
is a key weakness here. As shown in Table 2, the reanalysis does provide everything needed to compute 
total stratification

Papers:
Increasing ocean stratification over the past  half-century by Guancheng Li Nature Climate Change 2020.

Facts:
the Brunt–Väisälä frequency, or buoyancy frequency, is a measure of the stability of a fluid to vertical 
displacements such as those caused by convection

"""

def create_4d_src_dataset(source_4d_depth_levels, ds: xr.Dataset, pmid=None, name="temperature"
                          ) -> xr.Dataset:
    """Create dataset from N2 output

    Args:
        source_4d_depth_levels: the 4D N2 calculated array
        ds: The original dataset for temperature or salinity used to extract coordinates
        pmid: The new depth levels after N2 clauclations (one less than depth of temperature)
        name: Name of the variable to store as dataset

    Returns: Xarray Dataset containing N2 with coordinates

    """
    return xr.DataArray(
        name=name,
        data=source_4d_depth_levels,
        coords=[pmid,
                ds["time"].values,
                ds["lat"].values,
                ds["lon"].values],
        dims=["depth", "time", "lat", "lon"],
    ).to_dataset()


def extrapolate_biology(ds_in: xr.Dataset, ds_out: xr.Dataset) -> xr.Dataset:
    """Extrapolate/interpolate the biological data to the physical grid

    Args:
        ds_in: The biological data set (GOBH)
        ds_out: The physical (e.g. GLORYS12V1) dataset for temperature

    Returns: The biological dataset interpolated to the physical grid

    """
    ds_in=ds_in.chunk(chunks="auto")
    ds_out=ds_out.chunk(chunks="auto")
    
    regridder = xe.Regridder(
        ds_in,
        ds_out,
        "bilinear",
        extrap_method="inverse_dist",
        extrap_num_src_pnts=10,
        extrap_dist_exponent=2,
        unmapped_to_nan=True,
        ignore_degenerate=True,
    )
    return regridder(ds_in)


def merge_and_save_datasets(station: {}) -> {}:
    """Merges the various datasets in the dictionary and broadcasts
    Args:
        station: dictionary contining the various datasets

    Returns: dictionary with the new dataset as key and the old ones removed

    """
    ds_so = station["ds_so"]
    ds_thetao = station["ds_thetao"]
    ds_thetao = ds_thetao.assign(mask=ds_thetao["mask"].isel(time=0))
    ds_so = ds_so.assign(mask=ds_so["mask"].isel(time=0))

    ds_phyc = station["ds_phyc"]
    ds_phyc = ds_phyc.assign(mask=ds_phyc["mask"].isel(time=0))
    ds_phyc_extrapolated = extrapolate_biology(ds_in=ds_phyc, ds_out=ds_thetao)

    ds_chl = station["ds_chl"]
    ds_chl = ds_chl.assign(mask=ds_chl["mask"].isel(time=0))
    ds_chl_extrapolated = extrapolate_biology(ds_in=ds_chl, ds_out=ds_thetao)

    # Convert depth to pressure:
    # https://teos-10.github.io/GSW-Python/gsw_flat.html
    ds_thetao = ds_thetao.assign(pressure=gsw.p_from_z(ds_thetao["depth"], ds_thetao["lat"]))
    ds_thetao = ds_thetao.assign(depth_n=ds_thetao["depth"])
    ds_so = ds_so.assign(depth_n=ds_so["depth"])

    (ds1, ds2) = xr.broadcast(ds_so, ds_thetao)
    ds_temp_salt = xr.merge([ds1, ds2])
    station["ds_temp_salt"] = ds_temp_salt
    station["ds_phyc"] = ds_phyc_extrapolated
    station["ds_chl"] = ds_chl_extrapolated

    # Remove unneccessary dataset
    station.pop("ds_so")
    station.pop("ds_thetao")
    station.pop("ds_phyc")
    station.pop("ds_chl")

    netcdf_file1 = f"{station['base_directory']}/{station['station_name']}_temp_salt.nc"
    netcdf_file2 = f"{station['base_directory']}/{station['station_name']}_phyc.nc"
    netcdf_file3 = f"{station['base_directory']}/{station['station_name']}_chl.nc"

    for ds, netcdf_file in zip([ds_temp_salt, ds_phyc_extrapolated, ds_chl_extrapolated], [netcdf_file1, netcdf_file2, netcdf_file3]):
        if os.path.exists(netcdf_file):
            os.remove(netcdf_file)
        ds.mean({"lat","lon"}).to_netcdf(netcdf_file)
        print(f"Saved spatially averaged dataset to file: {netcdf_file}")
    return station


def calculate_brunt_vaisala(station: {}):
    """Calculate Bru-Vaisala frequency (stability) for the water column

    Args: salinity, temperature and pressure datasets stored in dictionary
        station: The station dictionary contining all required datasets
    """
    ds = station["ds_temp_salt"]
    ds = ds.transpose("depth", "time", "lat", "lon")

    N2, pmid = xr.apply_ufunc(gsw.Nsquared,
                              ds["so"].values,
                              ds["thetao"].values,
                              ds["pressure"].values,
                              input_core_dims=[["depth", "time", "lat", "lon"],
                                               ["depth", "time", "lat", "lon"],
                                               ["depth"]],
                              output_core_dims=[["depth", "time", "lat", "lon"],
                                                ["depth", "time", "lat", "lon"]],
                              exclude_dims=set(("depth", "time", "lat", "lon",)),
                              vectorize=True)

    ds_n2 = create_4d_src_dataset(N2, ds, pmid=np.squeeze(pmid[:, 0, 0, 0]), name="n2")

    base_n2 = f"{station['base_directory']}"
    netcdf_file = f"{base_n2}/{station['station_name']}_n2.nc"
    if os.path.exists(netcdf_file):
        os.remove(netcdf_file)
    ds_n2.to_netcdf(netcdf_file)
    print(f"Saved temperature, salinity, and phytoplankton to file: {netcdf_file}")


def organize_dataset(variable_id, station, start_time, end_time, first=False):
    baseURL = get_dataset_url(parameter=variable_id)
    print(f"Organizing GLORYS data at path {baseURL}")

    mapper = fs.get_mapper(
        get_dataset_url(parameter=variable_id)
    )

    consolidate_metadata(mapper)
    d = xr.open_zarr(mapper, consolidated=False)

    d = d.rename({"longitude": "lon", "latitude": "lat"})
    d["mask"] = create_land_ocean_mask(d)
    ds_masked = d.where(d.mask == 1, drop=True)

    st_lon = station["lon"]
    st_lat = station["lat"]

    if first or variable_id in ["phyc", "chl"]:
        list_of_lons, list_of_lats, list_of_distances = get_index_closest_points(
            st_lon, st_lat, ds_masked.lon, ds_masked.lat, ds_masked[variable_id].isel(time=0)
        )

        station["lons"] = list_of_lons
        station["lats"] = list_of_lats
    else:
        list_of_lons = station["lons"]
        list_of_lats = station["lats"]

    if list_of_lons:
        ds_interp = ds_masked.isel(
            lat=list_of_lats,
            lon=list_of_lons).resample(time="M").mean()

        ds_var = f"ds_{variable_id}"
        station[ds_var] = ds_interp

    return station


def get_dataset_url(parameter):
    try:
        return {
            "o2": f"gs://{ACTEA_bucket}/zarr/o2/",
            "no3": f"gs://{ACTEA_bucket}/zarr/no3/",
            "si": f"gs://{ACTEA_bucket}/zarr/si/",
            "so": f"gs://{ACTEA_bucket}/zarr/so/",
            "mlotst": f"gs://{ACTEA_bucket}/zarr/mlotst/",
            "thetao": f"gs://{ACTEA_bucket}/zarr/thetao/",
            "chl": f"gs://{ACTEA_bucket}/zarr/chl/",
            "intpp": f"gs://{ACTEA_bucket}/zarr/nppv/",
            "uo": f"gs://{ACTEA_bucket}/zarr/uo/",
            "vo": f"gs://{ACTEA_bucket}/zarr/vo/",
            "ph": f"gs://{ACTEA_bucket}/zarr/ph/",
            "spco2": f"gs://{ACTEA_bucket}/zarr/spco2/",
            "phyc": f"gs://{ACTEA_bucket}/zarr/phyc/",
            "siconc": f"gs://{ACTEA_bucket}/zarr/siconc/",
            "sithick": f"gs://{ACTEA_bucket}/zarr/sithick/",
        }[parameter]
    except Exception as e:
        return e



def create_land_ocean_mask(grid: xr.Dataset) -> xr.DataArray:
    """
    Function that creates land mask based on its longitude - latitude.
    Returns a DataArray to be included in a Dataset and used for extrapolation
    in xesmf.
    """
    print(f"Running create_land_ocean_mask {grid.sizes}")

    if grid.lon.ndim == 1:
        lon = grid.lon.values
        lat = grid.lat.values
    elif grid.lon.ndim == 2:
        lon = grid.lon[0, :].values
        lat = grid.lat[:, 0].values
    else:
        raise Exception(
            "Unable to understand dimensions of longitude/latitude: {}".format(
                grid.lon.ndim
            )
        )

    lon_180 = xr.where(lon > 180, lon - 360, lon)
    lon_grid, lat_grid = np.meshgrid(lon_180, lat)

    mask_data = global_land_mask.globe.is_ocean(lat_grid, lon_grid)

    return xr.DataArray(
        mask_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]
    ).astype(int)


def get_index_closest_points(
        st_lon: np.float,
        st_lat: np.float,
        lon: xr.DataArray,
        lat: xr.DataArray,
        da: xr.DataArray,

        max_distance_from_point=300
) -> ([], [], []):
    """
    Method that finds the n number of stations closest to a st_lon, st_lat point
    and checks to see if the values only contain nan (land) or if this is a valid
    ocean station.
    """
    from sklearn.neighbors import BallTree

    lons, lats = np.meshgrid(lon.values, lat.values)
    df = pd.DataFrame()
    df["lat"] = lats.ravel()
    df["lon"] = lons.ravel()

    # Setup Balltree using df as reference dataset
    # Use Haversine calculate distance between points on the earth from lat/long
    # haversine - https://pypi.org/project/haversine/
    tree = BallTree(np.deg2rad(df[["lat", "lon"]].values), metric="haversine")

    # Station
    df_st = pd.DataFrame()
    df_st["lat"] = st_lat
    df_st["lon"] = st_lon

    # Find the closest point in reference dataset for each in df_other
    # use k = 3 for 3 closest neighbors
    distances, indices = tree.query(np.deg2rad(np.c_[st_lat, st_lon]), k=100)

    list_of_lons = []
    list_of_lats = []
    list_of_distances = []
    r_km = 6371  # multiplier to convert to km (from unit distance)
    for d, ind in zip(distances, indices):
        print(f"\nFound closest matches to {st_lat:3.2f} and {st_lon:3.2f}:")
        for i, index in enumerate(ind):

            print(
                "== {:3.2f} {:3.2f} with distance {:.4f} km".format(
                    df["lat"][index], df["lon"][index], d[i] * r_km
                )
            )
            lon_index = np.where(lon == df["lon"][index])[0]
            lat_index = np.where(lat == df["lat"][index])[0]

            cont = False
            if var_name == "mlotst":
                if (not np.isnan(da[lat_index, lon_index])) and (d[i] * r_km < max_distance_from_point):
                    cont = True
            elif (not np.isnan(da[:, lat_index, lon_index]).all()) and (d[i] * r_km < max_distance_from_point):
                cont = True

            if cont:
                print("Using:")
                print(
                    "lon found at index {} lon {}".format(
                        lon_index[0], lon[lon_index[0]].values
                    )
                )
                print(
                    "lat found at index {} lat {}".format(
                        lat_index[0], lat[lat_index[0]].values
                    )
                )
                list_of_lons.append(lon_index[0])
                list_of_lats.append(lat_index[0])
                list_of_distances.append(d[i] * r_km)

    if list_of_lons:
        return list_of_lons, list_of_lats, list_of_distances
    return None, None, None


### MAIN
infile = "../data/sitelist_bs_20230110.csv"
df = pd.read_csv(infile, header=[0])
print(df.head())
base_directory = "../../shared/NSF"
if not os.path.exists(base_directory):
    os.makedirs(base_directory, exist_ok=True)

# Google Cloud client
ACTEA_bucket = "actea-shared"
fs = gcsfs.GCSFileSystem(project="downscale")
storage_client = storage.Client()
bucket = storage_client.bucket(ACTEA_bucket)

var_names = ["mlotst", "thetao", "so", "chl", "uo", "vo"]
var_names = ["thetao", "so", "phyc","chl"]
start_time = "1993-01-01"
end_time = "2000-12-31"
first = True
df_all_vars_stations = []

dfs_all = {}
stations = {}
dfs = []
maxindex = 90
# Prepare all the stations
for index, row in df.iterrows():
    
    if index > 16:
        st_lat = float(row.Latitude)
        st_lon = float(row.Longitude)

        station = {"station_name": row.Site,
                   "lat": st_lat,
                   "lon": st_lon,
                 #  "ID": row.ID,
                   "base_directory": base_directory}
        stations[row.Site] = station

        print(f"Station: {row.Site} latitude: {st_lat} longitude: {st_lon}")
        if index == maxindex:
            break

        for var_name in var_names:
            station = organize_dataset(
                var_name, station, start_time, end_time, first=first
            )
            print("Worked on station", station)
            first = False
        stations[row.Site] = station
        first = True

        station = merge_and_save_datasets(station)
        station = calculate_brunt_vaisala(station)
        stations[station] = station
