# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
import logging
import os

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as po
import pypsa
from _helpers import configure_logging
from add_electricity import load_costs
from pypsa.geo import haversine
from scipy.spatial import Voronoi
from shapely.geometry import LineString
from sklearn.neighbors import BallTree
from spopt.region import Spenc
from spopt.region.maxp import MaxPHeuristic

logger = logging.getLogger(__name__)


def consense(x):
    v = x.iat[0]
    assert (
        x == v
    ).all() or x.isnull().all(), "In {} cluster {} the values of attribute {} do not agree:\n{}".format(
        component, x.name, attr, x
    )
    return v


def center_of_mass(df, groupby=None, weight=None):
    df.geometry = df.to_crs(3035).centroid
    x = (
        df.groupby(groupby)
        .apply(lambda x: (x.geometry.x * x[weight]).sum() / x[weight].sum())
        .rename("x")
    )
    y = (
        df.groupby(groupby)
        .apply(lambda x: (x.geometry.y * x[weight]).sum() / x[weight].sum())
        .rename("y")
    )
    centers=gpd.GeoSeries(gpd.points_from_xy(x, y,crs=3035)).to_crs(4326)
    return pd.concat([centers.x, centers.y], axis=1, keys=["x", "y"])


def get_region_intersections(regions):
    intesections = dict()
    regions = regions.to_crs(3035).buffer(5000)
    for idx, region in regions.items():
        intesections.update({idx: regions[regions.intersects(region)].index.tolist()})
    return intesections


def move_generators(offshore_regions):
    '''
    This method attached the previously to onshore buses offshore generators to their offshore bus.

    Parameters
    ----------
    offshore_regions : GeoDataFrame
        GeoDataFrame containing all offshore regions.
    '''
    # Gets all offshore generators of the of the bus to which the offshore region is connected. Therefore, generators for which no offshore bus exists are included
    move_generators = (
        n.generators[n.generators.bus.isin(offshore_regions.bus.unique())]
        .filter(like="offwind", axis=0)
        .index.to_series()
        .str.replace(" offwind-\w+", "", regex=True)
    )

    # Add prefix to generator to know it is attached to an offshore bus
    prefix = "off_"
    move_generators = prefix + move_generators

    # Now only filter the offshore generators for which an offshore bus exists and move generators
    move_generators = move_generators[move_generators.isin(n.buses.index)]
    n.generators.loc[move_generators.index, "bus"] = move_generators

    # Only consider turbine cost and substation cost for offshore generators connected to offshore grid
    n.generators.loc[move_generators.index, "capital_cost"] = n.generators.loc[
        move_generators.index
    ].eval("turbine_cost + substation_cost")
    rename_index = dict(zip(move_generators.index, prefix + move_generators.index))
    n.generators.rename(index=rename_index, inplace=True)
    n.generators_t.p_max_pu.rename(columns=rename_index, inplace=True)


def add_links(df):
    # add lines only as DC links and don't consider AC anymore -> cost from DEA for AC links are currently taken but they are actually not tech specific

    # attach cable cost AC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = df["length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["HVDC submarine", "capital_cost"]
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.madd(
        "Link",
        names=df.index,
        carrier="DC",
        bus0=df["bus0"].values,
        bus1=df["bus1"].values,
        length=df["length"].values,
        capital_cost=cable_cost,
        underwater_fraction=1,
    )


def add_p2p_connections():
    # Creates a link between offshore generators and connected onshore buses as point to point connection instead of directly assigning the offshore generator to the onshore bus

    offshore_buses_name = n.buses[n.buses.index.str.contains("off")].index
    onshore_buses_name = offshore_regions.bus
    p2p_lines_df = pd.DataFrame(
        {"bus0": offshore_buses_name, "bus1": onshore_buses_name}
    ).reset_index(drop=True)
    p2p_lines_df.index = "off_p2p_" + p2p_lines_df.index.astype("str")
    p2p_lines_df.loc[:, "length"] = p2p_lines_df.apply(
        lambda x: haversine(
            n.buses.loc[x.bus0, ["x", "y"]], n.buses.loc[x.bus1, ["x", "y"]]
        ).item(),
        axis=1,
    )
    add_links(p2p_lines_df)


def add_offshore_bus_connections():
    # Create line for every offshore bus and connect it to onshore buses
    onshore_coords = n.buses.loc[offshore_regions.bus.unique(), ["x", "y"]]
    offshore_buses_coord = n.buses.loc[n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_hub_coord = n.buses.loc[n.buses.index.str.contains("hub"), ["x", "y"]]
    coords = pd.concat([onshore_coords, offshore_buses_coord, offshore_hub_coord])
    coords["xy"] = list(map(tuple, (coords[["x", "y"]]).values))

    # If no offshore exists, the offshore buses are interconnected, otherwise the buses are connected to the hubs and the hubs have an interconnection between the hubs
    if offshore_hub_coord.empty:
        offshore_coords = offshore_buses_coord
        onshore_connections = 1
    else:
        offshore_coords = offshore_hub_coord
        onshore_connections = 2
        hub_lines = pd.DataFrame({
            "bus0": "hub_"+ offshore_regions.hub.astype("str"),
            "bus1": "off_"+offshore_regions.index.to_series()
            }).reset_index(drop=True)
        hub_lines.loc[:, "length"] = hub_lines.apply(
            lambda x: haversine(coords.loc[x.bus0, "xy"], coords.loc[x.bus1, "xy"]).item(),
            axis=1,)
        hub_lines.index = "off_hub_" + hub_lines.index.astype("str")
        add_links(hub_lines)

    # Method to create evenly distributed connections between offshore busses or hub
    offshore_lines = create_meshed_grid(offshore_coords)

    # Connect every offshore bus / hub to at least #onshore_connections onshore bus
    onshore_lines = create_lines_kNN(offshore_coords, onshore_coords, k=onshore_connections)

    lines_df = pd.concat([offshore_lines, onshore_lines], axis=0, ignore_index=True)
    lines_df = lines_df.rename(
        columns={"source": "bus0", "target": "bus1", "weight": "length"}
    ).astype({"bus0": "string", "bus1": "string", "length": "float"})
    lines_df.loc[:, "length"] = lines_df.apply(
        lambda x: haversine(coords.loc[x.bus0, "xy"], coords.loc[x.bus1, "xy"]).item(),
        axis=1,
    )
    lines_df.drop(lines_df.query("length==0").index, inplace=True)
    lines_df.index = "off_" + lines_df.index.astype("str")
    add_links(lines_df)


def create_meshed_grid(buses):
    cells, generators = libpysal.cg.voronoi_frames(
        buses.values, clip="convex hull"
    )
    delaunay = libpysal.weights.Rook.from_dataframe(cells)
    line_graph = delaunay.to_networkx()
    line_graph = nx.relabel_nodes(
        line_graph, dict(zip(line_graph, buses.index))
    )
    lines = nx.to_pandas_edgelist(line_graph)
    # Remove lines of meshed grid which intersect with onshore shapes
    lines_filter = lines.apply(
    lambda x: LineString(
        [
            n.buses.loc[x.source, ["x", "y"]].astype("float"),
            n.buses.loc[x.target, ["x", "y"]].astype("float"),
        ]
    ),
    axis=1,
    )
    onshore_shape = onshore_regions.unary_union
    lines_filter = lines_filter.apply(lambda x: x.intersects(onshore_shape))
    lines.drop(lines[lines_filter].index, inplace=True)
    return lines

def create_lines_kNN(buses1, buses2, k=1):
    # Connect every buses1 to at least #onshore_connections onshore bus
    tree = BallTree(
        np.radians(buses1), leaf_size=40, metric="haversine"
    )
    _, ind = tree.query(np.radians(buses2), k=k)

    line_graph = nx.DiGraph()
    for i, bus in enumerate(buses2.index):
        for j in range(ind.shape[1]):
            bus1 = buses1.index[ind[i, j]]
            line_graph.add_edge(bus, bus1)

    return nx.to_pandas_edgelist(line_graph)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_offshore_grid", simpl="", clusters="64", offgrid="10"
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.clustered_network)

    offgrid = snakemake.wildcards["offgrid"]
    offgrid_config = snakemake.config["offshore_grid"]
    if not offgrid and not offgrid_config["p2p_connection"]:
        n.export_to_netcdf(snakemake.output[0])
    else:
        country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index(
            "name"
        )["geometry"]

        onshore_regions = gpd.read_file(snakemake.input.onshore_regions)

        offshore_regions = gpd.read_file(snakemake.input.offshore_regions)

        offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).set_index(
            "name"
        )["geometry"]

        Nyears = n.snapshot_weightings.objective.sum() / 8760.0

        costs = load_costs(
            snakemake.input.tech_costs,
            snakemake.config["costs"],
            snakemake.config["electricity"],
            Nyears,
        )

        offshore_generators = (
            n.generators.filter(regex="offwind", axis=0)
            .loc[:, ["p_nom_max", "bus"]]
            .copy()
        )
        offshore_generators["cf"] = (
            n.generators_t.p_max_pu.loc[:, offshore_generators.index]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            / 8760
        )
        offshore_generators["regions"] = offshore_generators.index.str.replace(
            " offwind-\w+", "", regex=True
        )
        offshore_generators = offshore_generators.groupby("regions").agg(
            {"p_nom_max": np.sum, "cf": np.mean, "bus": consense}
        )

        # offshore regions have more shapes than offshore generators have, don't know why, maybe rerun build renewable and add electricity and check again
        offshore_regions = offshore_regions.merge(
            offshore_generators, right_index=True, left_on="name"
        ).set_index("name")

        # calculate distance to offshore region
        coords = pd.DataFrame(index=offshore_regions.index)
        coords["onshore"] = list(
            map(tuple, (n.buses.loc[offshore_regions.bus, ["x", "y"]]).values)
        )
        coords["offshore"] = list(
            map(tuple, (offshore_regions[["x_region", "y_region"]]).values)
        )
        offshore_regions["distance"] = coords.apply(
            lambda x: haversine(x.onshore, x.offshore), axis=1
        )

        # only build grid for buses in country list and/or in sea shape
        countries = snakemake.config["offshore_grid"]["countries"]
        offshore_regions = offshore_regions.loc[
            offshore_regions.country.str.contains("|".join(countries))
        ]

        # if a shape for the offshore grid is selected only consider regions within this shape
        if snakemake.config["offshore_grid"]["sea_region"]:
            sea_shape = gpd.read_file(snakemake.config["offshore_grid"]["sea_region"])
            offshore_regions = offshore_regions[
                offshore_regions.intersects(sea_shape.geometry.unary_union)
            ]

        # only consider offshore regions which are bigger than 1GW and have a higher distance than 50m
        offshore_regions = offshore_regions.query("distance>=50 & p_nom_max>1000")

        offshore_regions["yield"] = offshore_regions.eval("p_nom_max * cf")

        # cluster buses to simplify grid or to get hubs
        if offgrid != "":
            n_clusters = int(offgrid)
            intesections = get_region_intersections(offshore_regions.reset_index())
            w = libpysal.weights.W(intesections)
            model = Spenc(
                offshore_regions,
                w,
                n_clusters=n_clusters,
                attrs_name=["yield"],
                gamma=0,
            )
            model.solve()
            offshore_regions["hub"] = np.array(model.labels_)
            hub_location = center_of_mass(
                offshore_regions, groupby="hub", weight="yield"
            )

            coords = coords.loc[offshore_regions.index, :]
            coords["hub"] = list(
                map(tuple, (hub_location.loc[offshore_regions.hub].values))
            )
            offshore_regions["distance_hub"] = coords.apply(
                lambda x: haversine(x.offshore, x.hub), axis=1
            )
            
            # if region closer to onshore, do not consider for offshore grid
            offshore_regions = offshore_regions[~(offshore_regions["distance_hub"] > offshore_regions["distance"])]

            n.madd(
                "Bus",
                names="hub_" + hub_location.index.astype("str").values,
                v_nom=220,
                x=hub_location["x"].values,
                y=hub_location["y"].values,
                substation_off=True,
            )
        
        # create busses for offshore regions
        n.madd(
                "Bus",
                names="off_" + offshore_regions.index,
                v_nom=220,
                x=offshore_regions["x_region"].values,
                y=offshore_regions["y_region"].values,
                substation_off=True,
                country=offshore_regions["country"].values,
            )
        # move offshore generators to offshore buses
        move_generators(offshore_regions)

        if offgrid_config["p2p_connection"]:
            add_p2p_connections()
        if offgrid != "":
            add_offshore_bus_connections()

        n.export_to_netcdf(snakemake.output[0])
