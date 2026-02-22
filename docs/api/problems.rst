.. _api_problems:

Problems — ``src.problems``
============================

Problem configuration classes define boundary conditions (fixed DOFs and
applied loads) for each test case.  All classes expose an ``apply(nodes)``
method that returns ``(loads, bcs)`` dictionaries.

.. contents:: On this page
   :local:

Factory
-------

.. automodule:: src.problems
   :members:
   :undoc-members:
   :show-inheritance:

TaggedProblem
--------------

Tag-based BC assignment from node tags embedded in the pipeline JSON.
This is the default problem type (``--problem tagged``).

.. automodule:: src.problems.tagged_problem
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

CantileverSetup
---------------

Standard cantilever: fixed at ``x = 0``, point load at ``x = nelx``.

.. automodule:: src.problems.cantilever
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

GenericProblem
---------------

Loads BC configuration from an external JSON file (``pipeline_bcs.json``).

.. automodule:: src.problems.generic
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
