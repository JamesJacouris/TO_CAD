.. _installation:

Installation
============

Requirements
------------

* Python 3.9 or later
* A Unix-like shell (macOS or Linux recommended; Windows via WSL2)
* `FreeCAD 0.21+ <https://www.freecad.org/>`_ (for CAD export only)

Python dependencies
-------------------

Install the core numerical stack::

   pip install numpy scipy numba matplotlib

Install the 3-D visualisation library::

   pip install open3d

Install the parameter-tuning library (optional)::

   pip install optuna

Install documentation dependencies::

   pip install sphinx sphinx-rtd-theme

No ``setup.py`` or ``pyproject.toml`` is provided yet; install in a virtual
environment to keep dependencies isolated::

   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install numpy scipy numba matplotlib open3d

Verifying the installation
--------------------------

Run a quick smoke test::

   python -c "
   import numpy, scipy, numba
   from src.optimization.top3d import Top3D
   from src.pipelines.baseline_yin.reconstruct import reconstruct_npz
   print('OK')
   "

If this prints ``OK`` the core pipeline is ready.

FreeCAD macro setup
-------------------

The FreeCAD export script (:mod:`src.export.freecad_reconstruct`) is a
**FreeCAD macro**, not a plain Python script.  To install:

1. Copy ``src/export/freecad_reconstruct.py`` to your FreeCAD macros folder
   (``~/.FreeCAD/Macro/`` on macOS/Linux).
2. In FreeCAD: **Macro → Macros…**, select ``freecad_reconstruct``, click
   **Execute**.
3. When prompted, point the dialog to your pipeline output ``.json`` file.
