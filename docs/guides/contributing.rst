.. _contributing:

Contributing
============

Branch policy
--------------

All active development happens on **Main_V2**.  Documentation is built from
this branch only.  Do not open PRs against legacy branches
(``Top3D_Yin_Pipeline_V2``, ``Top3D_Yin_Pipeline_Hybrid_Integrated``).

Code style
-----------

* PEP 8 formatting (``black`` recommended, line length 100).
* Google-style docstrings for all public functions::

      def foo(x: int) -> float:
          """One-line summary.

          Args:
              x: Description of x.

          Returns:
              Description of return value.
          """

* Numba-decorated functions (``@njit``) are exempt from docstring requirements
  due to limited autodoc support, but should have a plain comment block.

Adding a new problem
---------------------

1. Create ``src/problems/my_problem.py`` with a class ``MyProblem``.
2. Implement ``apply(nodes) -> tuple[dict, dict]`` — returns ``(loads, bcs)``.
3. Register it in :func:`src.problems.load_problem_config`.
4. Add an ``elif problem_name == 'my_problem':`` branch in ``run_top3d.py``
   if it requires custom Top3D BC setup.

Adding a new thinning mode
---------------------------

Thinning modes are integer constants passed to
:func:`src.pipelines.baseline_yin.thinning.thin_grid_yin`.  To add a new mode:

1. Add a new ``elif args.mode == N:`` branch inside ``thin_grid_yin``.
2. Define the candidate-selection and deletion logic.
3. Add tests in ``tests/test_thinning.py`` (if tests directory exists).

Building the docs
------------------

::

   cd docs
   pip install sphinx sphinx-rtd-theme
   make clean html
   open _build/html/index.html
