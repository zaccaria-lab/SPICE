"""Command-line entry point for SPICE.

Exposes the ``spice`` console command. It parses arguments, runs the main
``run_spice_bootstrap`` inference, writes all output tables to the chosen
output directory (via ``output_type='both'``) and produces the summary plots,
including the per-position copy-number probabilities plot.

Arguments are split into a lean set of *main* options aimed at typical users
and a ``dev`` subcommand holding advanced/tuning options, e.g.::

    spice profiles.tsv -o results/ --max-cn 8
    spice profiles.tsv -o results/ --max-cn 8 dev --phase-alleles --n-bootstraps 500

This module performs its work inside ``main()`` and is safe to import (no
side effects at import time), so it can be wired as a ``console_scripts``
entry point (``spice = "spice.bin.spice:main"``).
"""

import os
import sys
import argparse
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from spice.libs.spice_bootstrap import run_spice_bootstrap
import spice.libs.utils.logo as logo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_table(path):
    """Read a TSV/CSV table into a DataFrame, guessing the separator."""
    sep = '\t' if str(path).lower().endswith(('.tsv', '.tsv.gz', '.txt')) else ','
    return pd.read_csv(path, sep=sep)


def _get_root(edges):
    """Return the root node of a set of (Parent, Child) edges (parent that is
    never a child)."""
    parents = edges['Parent'].astype(str).unique()
    children = set(edges['Child'].astype(str).unique())
    roots = [p for p in parents if p not in children]
    return roots[0] if roots else str(parents[0])


def _check_gurobi():
    """Verify the Gurobi solver (and a usable license) are available.

    SPICE relies on a Gurobi-specific optimisation feature, so a free solver
    cannot be substituted. If Gurobi is missing or cannot be initialised, exit
    with an actionable message rather than failing deep inside the ILP.
    """
    try:
        import gurobipy as gp
    except Exception:
        sys.stderr.write(
            '\n[SPICE] ERROR: the Gurobi solver is required but was not found in this '
            'environment.\nSPICE uses a Gurobi-specific optimisation feature, so a free '
            'solver cannot be used instead.\n\n'
            'Install Gurobi into the current environment from its channel:\n\n'
            '    conda install -c gurobi gurobi\n\n'
            'or create an environment with SPICE and Gurobi together:\n\n'
            '    conda create -n spice -c conda-forge -c bioconda -c gurobi spice gurobi\n\n'
            'Gurobi also requires a valid license (free for academic use); set '
            'GRB_LICENSE_FILE to point to your license file.\n')
        sys.exit(1)
    try:
        env = gp.Env(empty=True)
        env.start()
        env.dispose()
    except Exception as e:
        sys.stderr.write(
            '\n[SPICE] ERROR: Gurobi is installed but could not be initialised, which is '
            'usually a missing or invalid license.\nObtain a free academic license from '
            'https://www.gurobi.com/academia/ and set GRB_LICENSE_FILE to point to it.\n'
            'Gurobi reported: {}\n'.format(e))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(args=None):
    """Parse command-line arguments and return a dict of run configuration.

    The returned dict contains the keyword arguments accepted by
    ``run_spice_bootstrap`` plus a few control keys used only by ``main()``
    (``data``, ``init_nodes_file``, ``make_plots`` and ``plot_solution_id``).
    """
    parser = argparse.ArgumentParser(
        prog='spice',
        description='SPICE: Subclone Probability Inference of Copy-number Evolution. '
                    'Reconstructs many equally-plausible copy-number phylogenies per '
                    'tumour to quantify uncertainty in each inferred CNA event.')

    # ---- Main arguments (non-expert facing) ----
    parser.add_argument(
        "DATA", type=str,
        help='Input table (TSV or CSV) of clone-specific copy-number profiles: one row per genomic '
             'segment per clone, with columns Node (clone id), Chr, Start, End, Allele (A/B; optional '
             'with --total-cn) and CN (integer copy number).')
    parser.add_argument(
        "-o", "--output", required=False, default='./spice_output', type=str,
        help='Output directory for result tables and plots, created if missing '
             '(default: ./spice_output).')
    parser.add_argument(
        "--total-cn", required=False, default=False, action='store_true',
        help='Infer on total copy number instead of allele-specific major/minor '
             'profiles (default: allele-specific).')
    parser.add_argument(
        "--max-cn", required=False, type=int, default=8,
        help='Maximum copy number modelled per segment; higher values are truncated '
             '(default: 8).')
    parser.add_argument(
        "--solution-pool-size", required=False, type=int, default=1000,
        help="Number of alternative, equally-plausible phylogenies to enumerate, shared "
             "between the topologies in proportion to their bootstrap frequency (SPICE's "
             "core uncertainty estimate); higher is more thorough but slower (default: 1000).")
    parser.add_argument(
        "--n-bootstraps", required=False, type=int, default=1000,
        help='Number of bootstrap replicates used to infer candidate tree topologies '
             '(default: 1000).')
    parser.add_argument(
        "-j", "--jobs", required=False, type=int, default=None,
        help='Number of parallel threads (default: all available cores; specifying a '
             'value explicitly is recommended).')
    parser.add_argument(
        "--time-limit", required=False, type=float, default=5.0,
        help='Maximum total runtime, in hours, for the ILP solving (default: 5).')
    parser.add_argument(
        "--seed", required=False, type=int, default=1,
        help='Random seed for reproducibility (default: 1).')
    parser.add_argument(
        "--verbose", required=False, default=False, action='store_true',
        help='Print detailed logging including Gurobi log, warnings, and errors.')

    # ---- Development arguments ----
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='Advanced/development options (see: spice DATA dev -h)')
    devparser = subparsers.add_parser('dev', description='Advanced and development options.')

    # Segment handling
    devparser.add_argument(
        "--min-seg-size", required=False, type=float, default=5e4,
        help='Segments smaller than this size (bp) are reconstructed with the simpler '
             'focal evolutionary model instead of the ILP (default: 50000).')
    devparser.add_argument(
        "--max-n-segs", required=False, type=int, default=None,
        help='Keep only the N largest segments in the ILP; remaining segments use the '
             'focal evolutionary model (default: none, segments are filtered by --min-seg-size).')
    devparser.add_argument(
        "--keep-consec-segs", required=False, default=False, action='store_true',
        help='Move small segments into the ILP so that every segment left in the focal evolutionary '
             'model has a neighbouring segment in the ILP (default: off).')
    devparser.add_argument(
        "--focal-max-cn", required=False, type=int, default=None,
        help='Maximum copy number for the focal evolutionary model (default: same as --max-cn).')
    devparser.add_argument(
        "--rm-small-segs", required=False, default=False, action='store_true',
        help='Drop small segments entirely instead of reconstructing them with the '
             'focal evolutionary model (default: off).')

    # Phasing
    devparser.add_argument(
        "--phase-alleles", required=False, default=False, action='store_true',
        help='Phase alleles before inference (CHISEL-style phasing) (default: off).')
    devparser.add_argument(
        "--leaf-independent-phasing", required=False, default=False, action='store_true',
        help='Phase each leaf independently rather than globally across all leaves '
             '(default: global phasing).')

    # Topology inference
    devparser.add_argument(
        "--topology-prop-limit", required=False, type=float, default=0.025,
        help='Minimum fraction of bootstrap trees supporting a topology for it to be '
             'kept (default: 0.025).')
    devparser.add_argument(
        "--topology-topn", required=False, type=int, default=10,
        help='Always keep the top-N most frequent topologies regardless of the '
             'proportion limit (default: 10).')
    devparser.add_argument(
        "--fixed-topology", required=False, type=str, default='False',
        help='Fix the tree topology instead of inferring it; provide the edges as a '
             'tuple of (parent, child) pairs (default: inferred).')
    devparser.add_argument(
        "--init-nodes", required=False, type=str, default=None,
        help='TSV of initial ancestral copy-number profiles (e.g. from MEDICC2) to '
             'warm-start the solver; requires --fixed-topology (default: none).')

    # Model / objective
    devparser.add_argument(
        "--max-cutoff", required=False, type=float, default=2.0,
        help='Enumerate only solutions whose cost is within this multiple of the '
             'optimal cost (default: 2).')
    devparser.add_argument(
        "--no-weight-wgd", required=False, default=False, action='store_true',
        help='Do not up-weight whole-genome-duplication (WGD) events in the ILP '
             'objective (default: WGD is weighted).')
    devparser.add_argument(
        "--wgd-fract-genome", required=False, type=float, default=0.5,
        help='Minimum fraction of an allele/genome that must be gained to call a WGD '
             'event (default: 0.5).')

    # Output / behaviour
    devparser.add_argument(
        "--single-solution", required=False, default=False, action='store_true',
        help='Return only the optimal solution of the lowest-cost topology instead of enumerating '
             'multiple solutions; NOTE: disables ranking and the probabilities plot (default: '
             'enumerate multiple solutions).')
    devparser.add_argument(
        "--no-rank", required=False, default=False, action='store_true',
        help='Skip probability computation and ranking of solutions; NOTE: disables the '
             'probabilities plot (default: rank).')
    devparser.add_argument(
        "--no-plot", required=False, default=False, action='store_true',
        help='Skip generating output plots (default: plots are produced).')
    devparser.add_argument(
        "--plot-solution-id", required=False, type=str, default=None,
        help='Solution (e.g. m2) whose copy-number profiles to plot, together with the '
             'copy-number probabilities of its topology (default: the top-ranked solution).')
    devparser.add_argument(
        "--devmode", "--debug", required=False, default=False, action='store_true',
        help='Output all internal ILP variables for debugging (default: off).')

    args = parser.parse_args(args)
    # Development args default to their values when the 'dev' subcommand is absent.
    devargs = args if args.subcommand == 'dev' else devparser.parse_args([])

    # ---- Validation ----
    if not os.path.isfile(args.DATA):
        raise ValueError('Input data file does not exist: {}'.format(args.DATA))
    if args.seed is not None and args.seed < 1:
        raise ValueError('The random seed must be positive!')
    if args.max_cn < 1:
        raise ValueError('--max-cn must be at least 1!')
    if args.solution_pool_size < 1:
        raise ValueError('--solution-pool-size must be at least 1!')
    if args.n_bootstraps < 1:
        raise ValueError('--n-bootstraps must be at least 1!')
    if args.time_limit <= 0:
        raise ValueError('--time-limit must be positive!')

    fixed_topology = eval(devargs.fixed_topology) if devargs.fixed_topology != 'False' else False
    if fixed_topology is not False and not isinstance(fixed_topology, tuple):
        raise ValueError('--fixed-topology must evaluate to a tuple of (parent, child) pairs.')
    if devargs.init_nodes is not None:
        if not os.path.isfile(devargs.init_nodes):
            raise ValueError('The --init-nodes file does not exist: {}'.format(devargs.init_nodes))
        if fixed_topology is False:
            raise ValueError('--init-nodes requires --fixed-topology to be provided.')

    # ---- Threads: default to all cores, but nudge the user to choose ----
    jobs = args.jobs
    if jobs is None:
        jobs = cpu_count()
        sys.stderr.write(
            '[SPICE] WARNING: --jobs was not specified; defaulting to all {} available '
            'cores. Please consider setting --jobs explicitly to a value matching the '
            'resources you want to use.\n'.format(jobs))
    elif jobs < 1:
        raise ValueError('--jobs must be at least 1!')

    multiple_solutions = not devargs.single_solution
    rank_multiple_trees = not devargs.no_rank
    make_plots = not devargs.no_plot
    if make_plots and (not multiple_solutions or not rank_multiple_trees):
        sys.stderr.write(
            '[SPICE] WARNING: the copy-number probabilities plot requires multiple '
            'solutions and ranking; it will be skipped because --single-solution '
            'and/or --no-rank were set.\n')

    return {
        # control keys (consumed by main, not passed to run_spice_bootstrap)
        "data": os.path.abspath(args.DATA),
        "init_nodes_file": devargs.init_nodes,
        "make_plots": make_plots,
        "plot_solution_id": devargs.plot_solution_id,
        # run_spice_bootstrap keyword arguments
        "min_seg_size": devargs.min_seg_size,
        "max_n_segs": devargs.max_n_segs,
        "keep_consec_segs": devargs.keep_consec_segs,
        "allele_specific": not args.total_cn,
        "phase_alleles": devargs.phase_alleles,
        "leaf_dependent_phasing": not devargs.leaf_independent_phasing,
        "max_cn": args.max_cn,
        "focal_max_cn": devargs.focal_max_cn,
        "verbose": args.verbose,
        "rs": args.seed,
        "n_threads": jobs,
        "time_limit": args.time_limit,
        "weight_wgd": not devargs.no_weight_wgd,
        "wgd_fract_genome": devargs.wgd_fract_genome,
        "multiple_solutions": multiple_solutions,
        "solution_pool_size": args.solution_pool_size,
        "max_cutoff": devargs.max_cutoff,
        "n_bootstraps": args.n_bootstraps,
        "fixed_topology": fixed_topology,
        "topology_prop_limit": devargs.topology_prop_limit,
        "topology_topn": devargs.topology_topn,
        "debug": devargs.devmode,
        "rm_small_segs": devargs.rm_small_segs,
        "rank_multiple_trees": rank_multiple_trees,
        "output_type": 'both',
        "output_dir": args.output,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_plots(res, output_dir, plot_solution_id=None):
    """Produce and save the SPICE summary plots into ``output_dir``.

    The profile plot shows solution ``plot_solution_id``, or if None, the
    top-ranked solution ('solution_rank' 1; the one with the lowest 'tree_id'
    if several are tied). The probabilities plot shows the topology of that
    solution. Without multiple solutions (--single-solution), the profile plot
    shows the optimal solution and the probabilities plot is skipped. If
    ``plot_solution_id`` is not found, neither plot is made.
    """
    # Non-interactive backend so plots render without a display (e.g. on clusters).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spice.libs.plotting.tree import plot_cn_tree, plot_cn_probabilities

    edges = res.get('total_edges_multisol')
    nodes = res.get('total_nodes_multisol')
    cn_probs = res.get('cn_probabilities')
    sol_probs = res.get('solution_probabilities')
    if edges is None or len(edges) == 0:
        # --single-solution: plot the optimal solution instead
        edges = res.get('total_edges_optimal')
        nodes = res.get('total_nodes_optimal')

    if edges is None or nodes is None or len(edges) == 0:
        sys.stderr.write('[SPICE] WARNING: no solutions available; skipping plots.\n')
        return
    if plot_solution_id is None:
        if sol_probs is not None and 'solution_rank' in sol_probs.columns:
            plot_solution_id = str(sol_probs.sort_values(['solution_rank', 'tree_id'])['Solution'].iloc[0])
        else:
            plot_solution_id = str(edges['Solution'].iloc[0])
    elif plot_solution_id not in edges['Solution'].astype(str).values:
        sys.stderr.write('[SPICE] WARNING: --plot-solution-id {} not found (see the Solution column of '
                         'spice_total_edges.tsv.gz); skipping plots.\n'.format(plot_solution_id))
        return
    plot_tree_id = edges.loc[edges['Solution'].astype(str) == plot_solution_id, 'tree_id'].iloc[0]

    # Copy-number probabilities plot (the required output), for the topology of the plotted solution.
    if cn_probs is not None and len(cn_probs) > 0:
        try:
            plot_cn_probabilities(
                edges, nodes, cn_probs, topology=plot_tree_id, leaf_order=None, color_mapping={},
                chrs='all', real_chr_lengths=True, plot_node_names=True, scale_x=1, scale_y=1)
            out = os.path.join(output_dir, 'spice_cn_probabilities.png')
            plt.gcf().savefig(out, dpi=150, bbox_inches='tight')
            plt.close('all')
            print('[SPICE] Saved copy-number probabilities plot (tree_id {}): {}'.format(plot_tree_id, out))
        except Exception as e:
            plt.close('all')
            sys.stderr.write('[SPICE] WARNING: could not produce probabilities plot: {}\n'.format(e))
    else:
        sys.stderr.write('[SPICE] WARNING: no probabilities available; skipping probabilities plot.\n')

    # Per-solution copy-number profile figure.
    try:
        plot_cn_tree(
            edges, nodes, leaf_order=None, cn_palette='RdBu_r', color_mapping={},
            plot_root=True, solution=plot_solution_id, chrs='all', real_chr_lengths=True)
        out = os.path.join(output_dir, 'spice_cn_profiles.png')
        plt.gcf().savefig(out, dpi=150, bbox_inches='tight')
        plt.close('all')
        print('[SPICE] Saved copy-number profile plot (solution {}): {}'.format(plot_solution_id, out))
    except Exception as e:
        plt.close('all')
        sys.stderr.write('[SPICE] WARNING: could not produce profile plot: {}\n'.format(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    print('\n======== SPICE: Subclone Probability Inference of Copy-number Evolution ========\n')
    if sys.stderr.isatty():
        sys.stderr.write(logo.render_logo())

    config = parse_args(args)
    data_path = config.pop('data')
    init_nodes_file = config.pop('init_nodes_file')
    make_plots = config.pop('make_plots')
    plot_solution_id = config.pop('plot_solution_id')

    if config['rs'] is not None:
        np.random.seed(config['rs'])

    print('-------- PARAMETERS --------')
    print('input : {}'.format(data_path))
    for k, v in config.items():
        print('{} : {}'.format(k, v))
    print('----------------------------\n')

    # Fail fast with an actionable message if the Gurobi solver is unavailable.
    _check_gurobi()

    os.makedirs(config['output_dir'], exist_ok=True)

    leaves_df = _read_table(data_path)
    initnodes = _read_table(init_nodes_file) if init_nodes_file is not None else None

    # Run inference. output_type='both' writes all result tables to output_dir
    # and returns them as a dict for plotting.
    res = run_spice_bootstrap(leaves_df, initnodes=initnodes, **config)

    if make_plots:
        _make_plots(res, config['output_dir'], plot_solution_id=plot_solution_id)

    print('\n======== COMPLETE ========\n')


if __name__ == '__main__':
    main()
