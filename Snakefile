# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

from os.path import normpath, exists
from shutil import copyfile, move, rmtree

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

from snakemake.utils import min_version

min_version("7.7")


if not exists("config.yaml"):
    copyfile("config.default.yaml", "config.yaml")


configfile: "config.yaml"


COSTS = f"data/costs_{config['costs']['year']}.csv"
ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

run = config.get("run", {})
RDIR = run["name"] + "/" if run.get("name") else ""
CDIR = RDIR if not run.get("shared_cutouts") else ""

LOGS = "logs/" + RDIR
BENCHMARKS = "benchmarks/" + RDIR
RESOURCES = "resources/" + RDIR if not run.get("shared_resources") else "resources/"
RESULTS = "results/" + RDIR


localrules:
    purge,


wildcard_constraints:
    simpl="[a-zA-Z0-9]*",
    clusters="[0-9]+m?|all",
    offgrid="\d+-[ch]|\d+-p-h|all|",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*",
    sector_opts="[-+a-zA-Z0-9\.\s]*",


include: "rules/common.smk"
include: "rules/collect.smk"
include: "rules/retrieve.smk"
include: "rules/build_electricity.smk"
include: "rules/build_sector.smk"
include: "rules/solve_electricity.smk"
include: "rules/postprocess.smk"
