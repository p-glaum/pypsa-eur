# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates Voronoi shapes for each bus representing both onshore and offshore
regions.

Relevant Settings
-----------------

.. code:: yaml

    countries:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``resources/regions_onshore.geojson``:

    .. image:: img/regions_onshore.png
        :scale: 33 %

- ``resources/regions_offshore.geojson``:

    .. image:: img/regions_offshore.png
        :scale: 33 %

Description
-----------
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import REGION_COLS, configure_logging
from pyproj import Geod
from scipy.spatial import Voronoi
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.validation import make_valid
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def calculate_area(shape, ellipsoid="WGS84"):
    """
    Calculates area in km².
    """
    geod = Geod(ellps=ellipsoid)
    return abs(geod.geometry_area_perimeter(shape)[0]) / 1e6


def find_closest_point():
    tree = BallTree(np.radians(buses1), leaf_size=40, metric="haversine")
    _, ind = tree.query(np.radians(buses2), k=k)


def transform_points(points, source="4326", target="3035"):
    points = gpd.GeoSeries.from_xy(points[:, 0], points[:, 1], crs=source).to_crs(
        target
    )
    points = np.asarray([points.x, points.y]).T
    return points


def check_validity(row):
    if not row.geometry.is_valid:
        logger.warning("Geometry not valid. Make geometry valid.")
        row.geometry = make_valid(row.geometry)
    return row


def cluster_points(n_clusters, point_list):
    """
    Clusters the inner points of a region into n_clusters.

    Parameters
    ----------
    n_clusters :
        Number of clusters
    point_list :
        List of inner points.

    Returns
    -------
        Returns list of cluster centers.
    """
    point_list = transform_points(np.array(point_list), source="4326", target="3035")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
        point_list
    )
    cluster_centers = transform_points(
        np.array(kmeans.cluster_centers_), source="3035", target="4326"
    )
    return cluster_centers


def fill_shape_with_points(shape, oversize_factor, num=10):
    """
    Fills the shape of the offshore region with points. This is needed for
    splitting the regions into smaller regions.

    Parameters
    ----------
    shape :
        Shape of the region.
    oversize_factor : int
        Factor by which the original region is oversized.
    num : int, optional
        Number of points added in the x and y direction.

    Returns
    -------
    inner_points :
        Returns a list of points lying inside the shape.
    """

    inner_points = list()
    x_min, y_min, x_max, y_max = shape.bounds
    iteration = 0
    while True:
        for x in np.linspace(x_min, x_max, num=num):
            for y in np.linspace(y_min, y_max, num=num):
                if Point(x, y).within(shape):
                    inner_points.append((x, y))
        if len(inner_points) > oversize_factor:
            break
        else:
            # perturb bounds that not the same points are added again
            num += 1
            x_min += abs(x_max - x_min) * 0.01
            x_max -= abs(x_max - x_min) * 0.01
            y_min += abs(y_max - y_min) * 0.01
            y_max -= abs(y_max - y_min) * 0.01
    return inner_points


def mesh_region(shape, n_regions):
    if n_regions == 1:
        shape = gpd.GeoDataFrame({"geometry": [shape]})
        return shape
    inner_points = fill_shape_with_points(shape, n_regions)
    cluster_centers = cluster_points(n_regions, inner_points)
    inner_regions = build_voronoi_cells(shape, cluster_centers)
    return inner_regions


def get_nearest_values(row, other_gdf):
    other_points = other_gdf.geometry
    nearest_geoms = nearest_points(row.geometry, other_points.unary_union)
    nearest_data = other_gdf.loc[other_gdf.geometry == nearest_geoms[1]]
    nearest_value = nearest_data.bus.values[0]

    return nearest_value


def build_voronoi_cells(shape, points):
    """
    Builds Voronoi cells from given points in the given shape.

    Parameters
    ----------
    shape :
        Shape where to build the cells in 4326 crs.
    points :
        List of points in 4326 crs.

    Returns
    -------
    split region
        Geopandas DataFrame containing the split regions.
    """
    cells = voronoi_partition_pts(points, shape)
    split_region = gpd.GeoDataFrame(
        {
            "geometry": cells,
        }
    )
    return split_region


def save_to_geojson(s, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    schema = {**gpd.io.file.infer_schema(s), "geometry": "Unknown"}
    s.to_file(fn, driver="GeoJSON", schema=schema)


def change_duplicate_names(group):
    if len(group) > 1:
        group["name"] = group["name"] + [f"_{i}" for i in range(len(group))]
    return group


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the polygon
    `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    # Convert shapes to equidistant projection shapes
    outline = gpd.GeoSeries(outline, crs="4326").to_crs("3035")[0]
    points = transform_points(points, source="4326", target="3035")

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            with np.errstate(invalid="ignore"):
                poly = poly.intersection(outline)

            polygons.append(poly)

        polygons = gpd.GeoSeries(polygons, crs="3035").to_crs(4326).values

    return polygons


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_bus_regions")
    configure_logging(snakemake)

    countries = snakemake.config["countries"]

    n = pypsa.Network(snakemake.input.base_network)

    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index("name")[
        "geometry"
    ]
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index("name")[
        "geometry"
    ]

    offshore_grid = snakemake.config["offshore_grid"]
    sea_shape = offshore_grid.get("sea_region", "")
    sea_shape = gpd.read_file(sea_shape) if sea_shape else ""

    split_offshore_regions = offshore_grid.get("split_offshore_regions", False)
    mesh_offshore_regions = offshore_grid.get("mesh_offshore_regions", True)
    threshold_area = float(offshore_grid.get("threshold_area", np.inf))
    threshold_length = 1000

    onshore_regions = []
    offshore_regions = []

    for country in countries:
        c_b = n.buses.country == country

        onshore_shape = country_shapes[country]
        onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
        onshore_regions.append(
            gpd.GeoDataFrame(
                {
                    "name": onshore_locs.index,
                    "x": onshore_locs["x"],
                    "y": onshore_locs["y"],
                    "geometry": voronoi_partition_pts(
                        onshore_locs.values, onshore_shape
                    ),
                    "country": country,
                }
            )
        )

        if country not in offshore_shapes.index:
            continue
        offshore_shape = offshore_shapes[country]
        offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
        offshore_regions_c = gpd.GeoDataFrame()
        if (
            mesh_offshore_regions
            and offshore_shape.intersects(sea_shape.geometry).any()
        ):
            region_to_mesh = offshore_shape.intersection(sea_shape.unary_union)
            offshore_shape = offshore_shape.difference(region_to_mesh)
            n_regions = int(np.ceil(calculate_area(region_to_mesh) / threshold_area))
            meshed_region = mesh_region(region_to_mesh, n_regions)
            locs = gpd.GeoDataFrame(
                {
                    "bus": offshore_locs.index,
                    "geometry": gpd.points_from_xy(
                        x=offshore_locs.x, y=offshore_locs.y
                    ),
                }
            )
            meshed_region["name"] = meshed_region.apply(
                get_nearest_values, other_gdf=locs, axis=1
            )
            meshed_region = meshed_region.merge(
                offshore_locs, left_on="name", right_index=True
            )
            meshed_region["country"] = country
            meshed_region.set_crs("4326", inplace=True)
            offshore_regions_c = pd.concat([offshore_regions_c, meshed_region])

        offshore_regions_c = pd.concat(
            [
                offshore_regions_c,
                gpd.GeoDataFrame(
                    {
                        "name": offshore_locs.index,
                        "x": offshore_locs["x"],
                        "y": offshore_locs["y"],
                        "geometry": voronoi_partition_pts(
                            offshore_locs.values, offshore_shape
                        ),
                        "country": country,
                    },
                    index=offshore_locs.index,
                    crs="4326",
                ),
            ]
        )
        offshore_regions_c.drop_duplicates(
            subset="geometry", inplace=True
        )  # some regions are duplicated
        if not offshore_regions_c.empty and split_offshore_regions:
            regions_oversize = offshore_regions_c.geometry.map(
                lambda x: calculate_area(x) / threshold_area
            )
            length_filter = (
                offshore_regions_c[regions_oversize < 1]
                .convex_hull.to_crs("3035")
                .length
                / 1000
                > threshold_length
            )
            regions_oversize.loc[length_filter[length_filter].index] = 2

            # only consider regions of interest for which the grid is build (could be made as additional argument)
            if isinstance(sea_shape, gpd.GeoDataFrame):
                regions_oversize[
                    ~offshore_regions_c.intersects(sea_shape.geometry.unary_union)
                ] = 0

            for bus, region in offshore_regions_c[regions_oversize > 1].iterrows():
                shape = region.geometry
                oversize_factor = regions_oversize.loc[bus]
                inner_regions = mesh_region(shape, int(np.ceil(oversize_factor)))
                inner_regions.set_index(
                    pd.Index([f"{bus}_{i}" for i in inner_regions.index], name="Bus"),
                    inplace=True,
                )
                inner_regions["name"] = inner_regions.index
                inner_regions["country"] = country
                inner_regions.loc[:, ["x", "y"]] = offshore_regions_c.loc[
                    bus, ["x", "y"]
                ].values
                offshore_regions_c = pd.concat(
                    [offshore_regions_c.drop(bus), inner_regions]
                )
        offshore_regions_c = offshore_regions_c.loc[
            offshore_regions_c.to_crs(3035).area > 50e6
        ]
        offshore_regions_c["area"] = offshore_regions_c.geometry.apply(
            lambda x: calculate_area(x)
        ).astype("float64")
        offshore_regions_c = offshore_regions_c.groupby("name", group_keys=False).apply(
            change_duplicate_names
        )
        offshore_regions.append(offshore_regions_c)

    pd.concat(onshore_regions, ignore_index=True).apply(check_validity, axis=1).to_file(
        snakemake.output.regions_onshore
    )
    if offshore_regions:
        offshore_regions = pd.concat(offshore_regions, ignore_index=True)
        centroid = offshore_regions.to_crs(3035).centroid.to_crs(4326)
        offshore_regions["x_region"] = centroid.x
        offshore_regions["y_region"] = centroid.y
        offshore_regions.apply(check_validity, axis=1).to_file(
            snakemake.output.regions_offshore
        )
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)
