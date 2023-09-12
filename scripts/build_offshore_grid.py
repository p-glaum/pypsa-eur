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
from sklearn.preprocessing import MinMaxScaler
from spopt.region import Spenc
from spopt.region.maxp import MaxPHeuristic

logger = logging.getLogger(__name__)


def normalize_series(series, min_=0, max_=1):
    scaler = MinMaxScaler((min_, max_))
    return scaler.fit_transform(series)


def consense(x):
    v = x.iat[0]
    assert (
        x == v
    ).all() or x.isnull().all(), "In {} cluster {} the values of attribute {} do not agree:\n{}".format(
        component, x.name, attr, x
    )
    return v


def center_of_mass(offshore_regions, groupby=None, weight=1):
    """
    Calculates the center of mass for a GeoDataFrame. E.g. used to find the
    center of a cluster.

    Parameters
    ----------
    offshore_regions : GeoDataFrame
    groupby : string
    weight : int, optional
        If 1, center is not weighted. If list of tuple, first entry specifies the weight of the column and the second the name of the column. If string it is weighted by the given column.

    Returns
    -------
    _type_
        _description_
    """
    df = offshore_regions.copy()
    df.geometry = df.to_crs(3035).centroid
    if weight != 1:
        if isinstance(weight, list):
            weights = 0
            for weight_entry in weight:
                weights += weight_entry[0] * normalize_series(
                    df.loc[:, [weight_entry[1]]]
                )
        elif isinstance(weight, str):
            weights = normalize_series(df.loc[:, [weight]])
    else:
        weights = weight
    df.loc[:, "weights"] = weights

    x = df.groupby(groupby).apply(lambda x: x.geometry.x @ x.weights / x.weights.sum())
    y = df.groupby(groupby).apply(lambda x: x.geometry.y @ x.weights / x.weights.sum())
    centers = gpd.GeoSeries(gpd.points_from_xy(x, y, crs=3035)).to_crs(4326)
    return pd.concat([centers.x, centers.y], axis=1, keys=["x", "y"])


def get_region_intersections(regions):
    intesections = dict()
    regions = regions.to_crs(3035).buffer(5000)
    for idx, region in regions.items():
        intesections.update({idx: regions[regions.intersects(region)].index.tolist()})
    return intesections


def move_generators(offshore_regions):
    """
    This method attached the previously to onshore buses offshore generators to
    their offshore bus.

    Parameters
    ----------
    offshore_regions : GeoDataFrame
        GeoDataFrame containing all offshore regions.
    """
    # Gets all offshore generators of the of the bus to which the offshore region is connected. Therefore, generators for which no offshore bus exists are included
    move_generators = (
        n.generators[n.generators.bus.isin(offshore_regions.bus.unique())]
        .filter(like="offwind", axis=0)
        .index.to_series()
        .str.replace(" offwind-\w+\s?\w+", "", regex=True)
    )

    # Add prefix to generator to know it is attached to an offshore bus
    prefix = "off_"
    move_generators = prefix + move_generators

    # Now only filter the offshore generators for which an offshore bus exists and move generators
    move_generators = move_generators[move_generators.isin(n.buses.index)]
    n.generators.loc[move_generators.index, "bus"] = move_generators

    # Only consider turbine cost and substation cost for offshore generators connected to offshore grid
    n.generators.loc[move_generators.index, "capital_cost"] = (
        n.generators.loc[move_generators.index, "turbine_cost"]
        + n.generators.loc[move_generators.index, "substation_cost"]
    )
    rename_index = dict(zip(move_generators.index, prefix + move_generators.index))
    n.generators.rename(index=rename_index, inplace=True)
    n.generators_t.p_max_pu.rename(columns=rename_index, inplace=True)


def add_links(df):
    n.madd(
        "Link",
        names=df.index,
        carrier="DC",
        bus0=df["bus0"].values,
        bus1=df["bus1"].values,
        length=df["length"].values,
        capital_cost=df["cost"].values,
        underwater_fraction=1,
        p_nom_extendable=True,
    )


def add_p2p_connections():
    # Creates a link between offshore generators and connected onshore buses as point to point connection instead of directly assigning the offshore generator to the onshore bus

    offshore_buses_name = n.buses.loc["off_" + offshore_regions.index].index
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
    # add lines only as DC links and don't consider AC anymore -> cost from DEA for AC links are currently taken but they are actually not tech specific
    line_length_factor = params["length_factor"]
    p2p_lines_df["cost"] = p2p_lines_df["length"].apply(
        lambda x: x * line_length_factor * costs.at["offshore-branch", "capital_cost"]
    )
    add_links(p2p_lines_df)


def add_offshore_bus_connections():
    # Create line for every offshore bus and connect it to onshore buses
    onshore_coords = n.buses.loc[offshore_regions.bus.unique(), ["x", "y"]]
    offshore_buses_coord = n.buses.loc[n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_hub_coord = n.buses.loc[n.buses.index.str.contains("hub"), ["x", "y"]]
    coords = pd.concat([onshore_coords, offshore_buses_coord, offshore_hub_coord])
    coords["xy"] = list(map(tuple, (coords[["x", "y"]]).values))
    line_length_factor = params["length_factor"]

    # If no offshore hub exists, the offshore buses are interconnected, otherwise the buses are connected to the hubs and the hubs have an interconnection between the hubs
    if offshore_hub_coord.empty:
        offshore_coords = offshore_buses_coord
        onshore_connections = 1
    else:
        offshore_coords = offshore_hub_coord
        onshore_connections = 2
        hub_lines = pd.DataFrame(
            {
                "bus0": "hub_" + offshore_regions.hub.astype("str"),
                "bus1": "off_" + offshore_regions.index.to_series(),
            }
        ).reset_index(drop=True)
        hub_lines.loc[:, "length"] = hub_lines.apply(
            lambda x: haversine(
                coords.loc[x.bus0, "xy"], coords.loc[x.bus1, "xy"]
            ).item(),
            axis=1,
        )
        hub_lines.index = "off_hub_" + hub_lines.index.astype("str")
        # add lines only as DC links and don't consider AC anymore -> cost from DEA for AC links are currently taken but they are actually not tech specific
        hub_lines["cost"] = hub_lines["length"].apply(
            lambda x: x
            * line_length_factor
            * costs.at["offshore-branch", "capital_cost"]
        )
        add_links(hub_lines)

    # Method to create evenly distributed connections between offshore buses or hub
    offshore_lines = create_meshed_grid(offshore_coords)

    # Connect every offshore bus / hub to at least #onshore_connections onshore bus
    onshore_lines = create_lines_kNN(
        offshore_coords, onshore_coords, k=onshore_connections
    )

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
    lines_df["cost"] = lines_df["length"].apply(
        lambda x: x * line_length_factor * costs.at["offshore-branch", "capital_cost"]
    )
    add_links(lines_df)


def create_meshed_grid(buses):
    cells, generators = libpysal.cg.voronoi_frames(buses.values, clip="convex hull")
    delaunay = libpysal.weights.Rook.from_dataframe(cells)
    line_graph = delaunay.to_networkx()
    line_graph = nx.relabel_nodes(line_graph, dict(zip(line_graph, buses.index)))
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
    tree = BallTree(np.radians(buses1), leaf_size=40, metric="haversine")
    _, ind = tree.query(np.radians(buses2), k=k)

    line_graph = nx.DiGraph()
    for i, bus in enumerate(buses2.index):
        for j in range(ind.shape[1]):
            bus1 = buses1.index[ind[i, j]]
            line_graph.add_edge(bus, bus1)

    return nx.to_pandas_edgelist(line_graph)


def add_wake_generators():
    mapping = (
        n.generators.filter(like="offwind", axis=0)
        .index.to_series()
        .str.replace(" offwind-\w+", "", regex=True)
    )
    wake_generators = n.generators.loc[
        mapping.index, :
    ]  # only consider offshore generators
    wake_generators = wake_generators[
        wake_generators.p_nom_max > 2e3
    ]  # only apply wake effect for generators greater than 2GW
    split_generators = split_generators = {
        2: wake_generators[wake_generators.p_nom_max <= 12e3],
        3: wake_generators[wake_generators.p_nom_max > 12e3],
    }
    factor_wake_losses = {
        1: 0,
        2: 0.1279732,
        3: 0.13902848,
    }  # factor for wake losses for each split generator
    max_capacity = {
        1: 2e3,
        2: 10e3,
        3: np.inf,
    }  # maximum capacity for each split generator

    generators_to_add = list()
    generators_t_to_add = list()
    generators_to_add_labels = list()
    generators_to_drop = list()

    # split generators into multiple generators with different time series
    for num, df in split_generators.items():
        for generator_i in df.index:
            generators_to_drop.append(generator_i)
            used_capacity = 0
            p_nom = 0
            for i in range(1, num + 1):
                generator = df.loc[generator_i].copy()
                generator_t = n.generators_t.p_max_pu.loc[:, generator_i].copy()
                if used_capacity + max_capacity[i] <= generator.p_nom_max:
                    generator["p_nom_max"] = max_capacity[i]
                    used_capacity += max_capacity[i]
                else:
                    generator.p_nom_max = generator.p_nom_max - used_capacity
                    used_capacity = generator.p_nom_max
                # adjust p_nom of the generators that the sum of the split generators is equal to the original p_nom
                if p_nom != generator.p_nom:
                    if max_capacity[i] < generator.p_nom - p_nom:
                        generator["p_nom"] = max_capacity[i]
                        generator["p_nom_min"] = max_capacity[i]
                    else:
                        generator["p_nom"] = generator["p_nom"] - p_nom
                        generator["p_nom_min"] = generator["p_nom_min"] - p_nom
                elif p_nom == generator["p_nom"]:
                    generator["p_nom"] = 0
                    generator["p_nom_min"] = 0
                p_nom += generator["p_nom"]
                generators_to_add_labels.append(generator_i + " w" + str(i))
                generators_to_add.append(generator)
                generators_t_to_add.append(generator_t * (1 - factor_wake_losses[i]))
    # delete original generators and add split generators
    n.generators.drop(index=generators_to_drop, inplace=True)
    n.generators_t.p_max_pu.drop(columns=generators_to_drop, inplace=True)
    # add wake effect generators
    n.generators = pd.concat(
        [
            n.generators,
            pd.concat(
                generators_to_add, axis=1, keys=generators_to_add_labels
            ).T.infer_objects(),
        ],
        axis=0,
    )
    n.generators_t.p_max_pu = pd.concat(
        [
            n.generators_t.p_max_pu,
            pd.concat(generators_t_to_add, axis=1, keys=generators_to_add_labels),
        ],
        axis=1,
    )
    n.generators_t.p_max_pu.columns.names = ["Generator"]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_offshore_grid", simpl="", clusters="64", offgrid="all-wake"
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.clustered_network)
    offgrid = snakemake.wildcards["offgrid"].split("-")[0]
    wake_effect = "wake" in snakemake.wildcards["offgrid"]

    params = snakemake.params
    offgrid_config = params["offgrid"]
    if (
        not offgrid
        and not offgrid_config["p2p_connection"]
        and not offgrid_config["wake_effect"]
    ):
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
            params.costs,
            params["max_hours"],
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
            lambda x: haversine(x.onshore, x.offshore).flatten().item(), axis=1
        )

        # only build grid for buses in country list and/or in sea shape
        countries = offgrid_config["countries"]
        offshore_regions = offshore_regions.loc[
            offshore_regions.country.str.contains("|".join(countries))
        ]

        # if a shape for the offshore grid is selected only consider regions within this shape
        if offgrid_config["sea_region"]:
            sea_shape = gpd.read_file(offgrid_config["sea_region"])
            offshore_regions = offshore_regions[
                offshore_regions.intersects(sea_shape.geometry.unary_union)
            ]

        # model wake effect for offshore generators in relevant offshore region
        if wake_effect:
            add_wake_generators()

        # only consider offshore regions which are bigger than 1GW and have a higher distance than 50m
        offshore_regions = offshore_regions.query("distance>=50 & p_nom_max>1000")

        offshore_regions["yield"] = offshore_regions.eval("p_nom_max * cf")

        # create buses for offshore regions
        n.madd(
            "Bus",
            names="off_" + offshore_regions.index,
            v_nom=220,
            x=offshore_regions["x_region"].values,
            y=offshore_regions["y_region"].values,
            substation_off=True,
            country=offshore_regions["country"].values,
        )

        if offgrid_config["p2p_connection"]:
            add_p2p_connections()

        # cluster buses to simplify grid or to get hubs
        if offgrid.isnumeric():
            n_clusters = int(offgrid)
            intesections = get_region_intersections(offshore_regions.reset_index())
            w = libpysal.weights.W(intesections)
            # p_nom_max = int(offgrid.split("-")[0]) * 1e3
            # model = MaxPHeuristic(
            #     offshore_regions,
            #     w,
            #     attrs_name=["yield"],
            #     threshold_name="p_nom_max",
            #     threshold=20e3,
            # )
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
                offshore_regions,
                groupby="hub",
                weight=[(0.2, "yield"), (0.8, "distance")],
            )

            coords = coords.loc[offshore_regions.index, :]
            coords["hub"] = list(
                map(tuple, (hub_location.loc[offshore_regions.hub].values))
            )
            offshore_regions["distance_hub"] = coords.apply(
                lambda x: haversine(x.offshore, x.hub), axis=1
            )

            # ax = offshore_regions.plot(column='hub', categorical=True, edgecolor='w', legend=True, cmap="tab20c", figsize=(20,20))
            # ax.scatter(hub_location["x"], hub_location["y"])
            # if region closer to onshore, do not consider for offshore grid
            offshore_regions = offshore_regions[
                ~(offshore_regions["distance_hub"] > offshore_regions["distance"])
            ]

            n.madd(
                "Bus",
                names="hub_" + hub_location.index.astype("str").values,
                v_nom=220,
                x=hub_location["x"].values,
                y=hub_location["y"].values,
                substation_off=True,
            )

        # move offshore generators to offshore buses
        move_generators(offshore_regions)

        if offgrid != "":
            add_offshore_bus_connections()

        n.export_to_netcdf(snakemake.output[0])
