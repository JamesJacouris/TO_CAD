TO_CAD Pipeline Documentation
==============================

**TO_CAD** converts 3-D topology-optimised density fields into clean beam-and-plate
CAD skeletons ready for FreeCAD.  The pipeline chains four stages:

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Stage
     - Script / Module
     - What it does
   * - 0
     - ``run_top3d.py``
     - 3-D topology optimisation (Python Top3D) → density ``.npz``
   * - 1
     - :mod:`src.pipelines.baseline_yin.reconstruct`
     - Yin medial-axis thinning → beam graph + plate regions → ``.json``
   * - 2
     - :mod:`src.optimization.size_opt`
     - Optimality-Criteria cross-section sizing
   * - 3
     - :mod:`src.optimization.layout_opt`
     - L-BFGS-B node-position layout optimisation

The final JSON is imported into FreeCAD via
:mod:`src.export.freecad_reconstruct`.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   guides/installation
   guides/quickstart
   guides/pipeline_walkthrough
   guides/freecad_export

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/data_flow
   architecture/algorithms
   architecture/json_schema

.. toctree::
   :maxdepth: 1
   :caption: CLI Reference

   cli/run_pipeline
   cli/run_top3d
   cli/tune_parameters

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/pipelines
   api/optimization
   api/problems
   api/curves
   api/export
   api/tuning

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   guides/contributing
   guides/testing

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
