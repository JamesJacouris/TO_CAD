.. _architecture_overview:

Architecture Overview
======================

.. contents:: On this page
   :local:
   :depth: 2

Repository layout
------------------

::

   TO_CAD/
   ├── run_pipeline.py            # Main orchestrator (CLI + Python API)
   ├── run_top3d.py               # Stage 0: topology optimisation
   ├── run_top3d_rocker_arm.py    # Rocker-arm specific Top3D script
   ├── tune_parameters.py         # Optuna parameter search
   │
   ├── src/
   │   ├── optimization/
   │   │   ├── top3d.py           # Python Top3D solver
   │   │   ├── fem.py             # 3-D Euler-Bernoulli FEA
   │   │   ├── size_opt.py        # Optimality-Criteria radius sizing
   │   │   └── layout_opt.py      # L-BFGS-B node-position optimisation
   │   │
   │   ├── pipelines/
   │   │   └── baseline_yin/
   │   │       ├── reconstruct.py      # Stage 1 orchestrator + JSON export
   │   │       ├── thinning.py         # Yin medial-axis thinning
   │   │       ├── topology.py         # Topological predicates (Numba)
   │   │       ├── graph.py            # Skeleton → graph extraction
   │   │       ├── postprocessing.py   # Graph pruning / cleanup
   │   │       ├── plate_extraction.py # Plate region recovery (hybrid)
   │   │       ├── joint_creation.py   # Beam-plate joint snapping
   │   │       ├── surface_fitting.py  # B-spline plate mid-surfaces
   │   │       └── visualization.py    # Open3D / Matplotlib visualisation
   │   │
   │   ├── problems/
   │   │   ├── __init__.py        # load_problem_config() factory
   │   │   ├── cantilever.py
   │   │   ├── tagged_problem.py  # Tag-based BC propagation
   │   │   └── generic.py
   │   │
   │   ├── curves/
   │   │   └── spline.py          # Cubic Bézier fitting & sampling
   │   │
   │   ├── export/
   │   │   └── freecad_reconstruct.py  # FreeCAD macro
   │   │
   │   └── tuning/
   │       ├── pipeline_runner.py  # Subprocess wrapper for trials
   │       └── metrics.py          # Geometry fidelity metrics
   │
   └── docs/                       # Sphinx documentation (this site)

Module dependency graph
------------------------

.. graphviz::
   :caption: Pipeline module dependencies (dashed edges = hybrid-only paths)

   digraph pipeline {
       rankdir=LR;
       node [shape=box, fontname="Helvetica", fontsize=10, style=filled, fillcolor="#f0f4ff"];
       edge [fontsize=9];

       // Top-level orchestrators
       run_pipeline [label="run_pipeline.py", fillcolor="#d0e8ff", shape=ellipse];
       run_top3d   [label="run_top3d.py",    fillcolor="#d0e8ff", shape=ellipse];

       // Stage 0
       top3d [label="optimization.top3d"];

       // Stage 1 — reconstruction
       reconstruct [label="baseline_yin.reconstruct"];
       thinning    [label="baseline_yin.thinning"];
       topology    [label="baseline_yin.topology"];
       graph_mod   [label="baseline_yin.graph"];
       postproc    [label="baseline_yin.postprocessing"];
       plate_ext   [label="baseline_yin.plate_extraction\n[hybrid]", fillcolor="#e8ffe8"];
       joint_cr    [label="baseline_yin.joint_creation\n[hybrid]",   fillcolor="#e8ffe8"];
       surf_fit    [label="baseline_yin.surface_fitting"];
       viz         [label="baseline_yin.visualization"];

       // Stage 2 / 3 — optimisation
       size_opt    [label="optimization.size_opt"];
       layout_opt  [label="optimization.layout_opt"];
       fem         [label="optimization.fem"];

       // Problems
       problems    [label="problems\n(load_problem_config)"];

       // Edges
       run_pipeline -> run_top3d;
       run_top3d    -> top3d;
       run_pipeline -> reconstruct;
       reconstruct  -> thinning;
       thinning     -> topology;
       reconstruct  -> graph_mod;
       reconstruct  -> postproc;
       reconstruct  -> plate_ext [style=dashed];
       reconstruct  -> joint_cr  [style=dashed];
       reconstruct  -> surf_fit;
       reconstruct  -> viz;
       run_pipeline -> size_opt;
       run_pipeline -> layout_opt;
       size_opt     -> fem;
       layout_opt   -> fem;
       run_pipeline -> problems;
       size_opt     -> problems;
       layout_opt   -> problems;
   }

Design decisions
-----------------

**Classify before thinning** (hybrid mode)
   Zone classification is applied *after* a single hybrid thinning pass
   (``mode=3``), not to the original solid.  This avoids the ambiguity of
   classifying thick solid blobs.

**Separate JSON edge format** ``[u, v, r]``
   All JSON edge lists use exactly three elements: node index ``u``, node
   index ``v``, radius ``r`` (mm).  Internal pipeline edge tuples carry
   additional data ``[u, v, weight, pts, r]`` but are converted on export.

**Direct Python API for Stage 1**
   :func:`src.pipelines.baseline_yin.reconstruct.reconstruct_npz` is a clean
   Python-callable API.  ``run_pipeline.py`` calls it directly (no subprocess)
   so exceptions propagate cleanly and there is no argument serialisation cost.

**FEM uses straight beams always**
   Even when ``--curved`` is specified, finite-element analysis always uses
   straight Euler-Bernoulli elements.  Bézier curves are purely geometric /
   visualisation output; they do not influence structural results.
