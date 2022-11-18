# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
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
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``resources/regions_onshore.geojson``:

    .. image:: ../img/regions_onshore.png
        :scale: 33 %

- ``resources/regions_offshore.geojson``:

    .. image:: ../img/regions_offshore.png
        :scale: 33 %

Description
-----------
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import REGION_COLS, configure_logging
from pyproj import Geod
from scipy.spatial import Voronoi
from shapely import affinity
from shapely.geometry import Point, Polygon
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def calculate_area(shape, ellipsoid="WGS84"):
    geod = Geod(ellps=ellipsoid)
    return abs(geod.geometry_area_perimeter(shape)[0]) / 1e6


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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(point_list)
    return kmeans.cluster_centers_


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


def build_voronoi_cells(shape, points):
    """
    Builds Voronoi cells from given points in the given shape.

    Parameters
    ----------
    shape :
        Shape where to build the cells.
    points :
        List of points.

    Returns
    -------
    split region
        Geopandas DataFrame containing the split regions.
    """
    split_region = gpd.GeoDataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "geometry": voronoi_partition_pts(points, shape),
        }
    )
    return split_region


def save_to_geojson(s, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    schema = {**gpd.io.file.infer_schema(s), "geometry": "Unknown"}
    s.to_file(fn, driver="GeoJSON", schema=schema)


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py
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
    points = gpd.GeoSeries.from_xy(points[:, 0], points[:, 1], crs="4326").to_crs(
        "3035"
    )
    points = np.asarray([points.x, points.y]).T

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

            poly = poly.intersection(outline)

            polygons.append(poly)

        polygons = gpd.GeoSeries(polygons, crs="3035").to_crs(4326).values

    # throws error if converted to np.array because multipolygons are split into polygons
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
        offshore_regions_c = gpd.GeoDataFrame(
            {
                "name": offshore_locs.index,
                "x": offshore_locs["x"],
                "y": offshore_locs["y"],
                "geometry": voronoi_partition_pts(offshore_locs.values, offshore_shape),
                "country": country,
            },
            index=offshore_locs.index,
        )
        offshore_regions_c = offshore_regions_c.loc[offshore_regions_c.area > 1e-2]
        offshore_regions_c.loc[:, "x"] = offshore_regions_c.centroid.x.values
        offshore_regions_c.loc[:, "y"] = offshore_regions_c.centroid.y.values
        split_offshore_regions = snakemake.config["enable"].get(
            "split_offshore_regions", False
        )
        offshore_regions_c.drop_duplicates(
            subset="geometry", inplace=True
        )  # some regions are duplicated
        if not offshore_regions_c.empty and split_offshore_regions:
            threshold_area = 15000  # km2 threshold at which regions are split
            threshold_length = (
                10  # to split very long regions with area less than 15000 km2
            )
            region_oversize = offshore_regions_c.geometry.map(
                lambda x: calculate_area(x) / threshold_area
            )
            length_filter = (
                offshore_regions_c[region_oversize < 1].geometry.length
                > threshold_length
            )
            region_oversize.loc[length_filter[length_filter].index] = 2

            for bus, region in offshore_regions_c[region_oversize > 1].iterrows():
                shape = region.geometry
                oversize_factor = region_oversize.loc[bus]
                inner_points = fill_shape_with_points(shape, oversize_factor)
                cluster_centers = cluster_points(
                    int(np.ceil(oversize_factor)), inner_points
                )
                inner_regions = build_voronoi_cells(shape, cluster_centers)
                inner_regions.set_index(
                    pd.Index([f"{bus}_{i}" for i in inner_regions.index], name="Bus"),
                    inplace=True,
                )
                inner_regions["name"] = inner_regions.index
                inner_regions["country"] = country
                offshore_regions_c = pd.concat(
                    [offshore_regions_c.drop(bus), inner_regions]
                )
        offshore_regions_c["area"] = offshore_regions_c.geometry.apply(
            lambda x: calculate_area(x)
        ).astype("float64")
        offshore_regions.append(offshore_regions_c)

    pd.concat(onshore_regions, ignore_index=True).to_file(
        snakemake.output.regions_onshore
    )
    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(
            snakemake.output.regions_offshore
        )
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)
