# Project Logbook

**Student:** James Jacouris
**Project:** Automated Conversion of Topology-Optimised Meshes to Parametric CAD Models
**Supervisor:** Dr Zahur Ullar
**Module:** ENGI 4093 — L4 Dissertation Project
**Academic Year:** 2025–2026

---

## Week 1 — 6–12 October 2025

### Supervisor Meeting (8 Oct)

**Questions Prepared:**
- Is there any existing work from Durham students or other researchers that would help define the starting point?
- What software is recommended?
- What are the expected goals, objectives, and deliverables?

**Discussion Notes:**
- The project is grounded in density-based topology optimisation (TO). The core challenges were identified:
  1. How to extract boundaries from the density field — *geometry extraction*
  2. How to convert extracted geometry into a parametric CAD model using b-splines
- Supervisor emphasised that the density field does not have a well-defined boundary, so thresholding and smoothing strategies will be critical.

**Actions Agreed:**
- Research existing commercial software (Solidworks, Fusion 360, Abaqus) — what TO capabilities they offer and current limitations
- Explore open-source tools: FreeTO, Top3DSTL, MATLAB Top3D code
- Study adaptive b-splines and density-based optimisation methods
- Complete risk assessment and ethics form

### Lecture — Dissertation Overview (10 Oct)
- Attended project overview lecture. Key takeaways: logbook must be submitted as PDF at end of project; the logbook itself is not assessed but supports academic integrity and demonstrates the working process.

### Research Plan — Structured Approach
Set up a structured initial research plan with four defined tasks:
1. **Understand Topology Optimisation** — Review SIMP, level-set, and homogenisation methods. Understand density-based TO outputs and their limitations.
2. **Compare Reconstruction Methods** — Survey existing approaches for converting TO output to usable geometry.
3. **Study Outputs & Limitations** — What does each method actually produce? Jagged meshes, smooth surfaces, parametric models?
4. **Review Prior TO→CAD Work** — Focus on published efforts to bridge the TO-CAD gap: spline patches, medial-axis methods, skeletal frameworks.

Deliverable from this phase: structured literature review chapter.

### Pre-Meeting Research Document — Foundational Study
Before the first supervisor meeting, compiled a comprehensive research document covering:

**TO Fundamentals:**
- SIMP (Solid Isotropic Material with Penalisation) — the most established density-based method; penalises intermediate densities to drive toward 0/1 distribution
- Level-set methods — define boundaries implicitly as zero-contour of a level-set function; cleaner boundaries but computationally expensive
- Moving Morphable Components (MMC) — parametric alternative using moveable geometric components as design variables

**Mesh vs CAD — The Core Problem:**
Identified four key differences between TO mesh output and parametric CAD:
1. TO produces density fields or triangulated meshes — not B-rep geometry
2. Mesh vertices have no parametric meaning — cannot be edited by designers
3. Feature recognition (holes, fillets, chamfers) is lost in the mesh representation
4. Manufacturing constraints (wall thickness, overhang angles) are not encoded

**B-spline/NURBS Mathematics:**
- Studied how Non-Uniform Rational B-Splines (NURBS) form the mathematical foundation of modern CAD
- B-spline curves defined by control points, knot vectors, and basis functions
- Trimmed NURBS surfaces enable complex CAD shapes through boundary representation (B-rep)
- Constructive Solid Geometry (CSG) recursively combines primitive shapes using boolean operations, stored as a CSG tree

**Existing Approaches Reviewed (6 papers in detail):**
- **Larsen & Jensen (2009)** — early work on automated NURBS fitting to TO results
- **Fang et al. (2017)** — curvature-based mesh segmentation for patch fitting
- **Polak (2023)** — skeleton extraction with graph simplification for design reconstruction
- **Zhang et al. (2025)** — recent neural skeleton approaches combining skeleton priors with implicit surfaces
- **Wang et al. (2023)** — parametric/isogeometric TO approaches integrating CAD directly into the optimisation loop

**Challenges Identified:** (1) preserving topology during conversion, (2) handling branching junctions, (3) achieving sufficient surface smoothness, (4) maintaining structural fidelity, (5) producing genuinely parametric (editable) output.

**Seven Potential Directions for Project:**
1. Reproduce Yin's homotopic skeletonisation pipeline
2. Build an integrated single-input/single-output tool
3. Extend to surface-based reconstruction (lofted surfaces)
4. Incorporate manufacturability constraints
5. Enable parametric reconstruction for downstream CAD editing
6. Validate via FEA comparison (original TO mesh vs reconstructed geometry)
7. Develop user-friendly interface with recommended parameter settings

### Weekly Summary
**Focus:** Project orientation and initial scoping.
**Work Completed:** Set up Notion workspace for project tracking. Created initial project structure with sections for meetings, research, and technical notes. Began literature search on existing TO-to-CAD conversion methods. Compiled extensive pre-meeting research document covering TO fundamentals, the mesh-CAD gap, B-spline/NURBS mathematics, six existing approaches, identified challenges, and potential project directions.
**Key Achievements:** Clear identification of the two core research challenges (geometry extraction and parametric conversion). Comprehensive foundational research document compiled. Deliverable planned: structured literature review chapter.
**Reflection:** The problem is broader than I initially expected — it spans computational geometry, optimisation theory, and CAD kernel APIs. Need to narrow scope quickly to avoid spreading too thin. The pre-meeting research gave me a strong foundation and helped frame good questions for the first supervisor meeting.

---

## Week 2 — 13–19 October 2025

### Supervisor Meeting (14 Oct)

**Discussion Notes:**
- Discussed geometry engines that could form the backend of a conversion tool: OpenCascade (open-source), Cubit (meshing), ParaView/Parasolid (visualisation/modelling)
- Supervisor suggested looking at how existing workflows handle the TO → physical part pipeline

**Actions Agreed:**
- Use Solidworks to walk through the existing manual workflow: run TO, export mesh, manually reconstruct in CAD
- Search for additional literature on automated conversion methods
- Explore alternative software tools
- Attempt to replicate at least one existing method from the literature

### Solidworks TO Experiment
Ran topology optimisation in Solidworks on a bracket under pressure loading. The TO output was a triangulated mesh that could not be directly edited or modified parametrically — confirming firsthand the core problem this project addresses. Even a simple bracket required significant manual effort to reconstruct as an editable CAD part.

### Development Environment Setup (14 Oct)
- Set up VS Code on Mac for Python CAD operations
- Created GitHub repository for version control
- Established project folder structure for code, data, and documentation

### Literature Survey — Technique Categorisation
Conducted systematic categorisation of all reconstruction techniques found in literature into four families:
1. **Skeletal-based** — Medial axis extraction → skeleton graph → parametric primitives (cylinders, spheres). Topology-preserving, interpretable. Core references: Yin & Cirak (2019, 2020), Polak (2023).
2. **Volume-based** — Direct voxel manipulation, density field processing. Simple but no parametric output.
3. **Surface-based** — NURBS/B-spline patch fitting to mesh boundary. Smooth results but poor topology handling, often requires manual segmentation.
4. **Feature-based** — Recognise geometric features (edges, holes, fillets) and reconstruct from recognised primitives. Promising for manufacturing but requires robust feature detection.

### AMRTO Code Implementation Attempt
Found and attempted to implement the AMRTO (Adaptive Mesh Refinement for TO) code from GitHub. Useful for understanding how existing academic pipelines handle the TO-to-geometry conversion, though the code had significant setup complexity.

### Instant Meshes Troubleshooting
Spent approximately 3 hours getting Instant Meshes (an open-source remeshing tool) working on macOS. The issue was macOS privacy settings blocking the unsigned application. Eventually resolved by:
- Modifying macOS security settings to allow unsigned applications
- Running a bash fix script to clear quarantine attributes
- Tool proved useful for understanding remeshing as a post-processing step for TO output

### MMC TO Research
Investigated Moving Morphable Components (MMC) as an alternative TO formulation. MMC uses parametric geometric components as design variables, which naturally produces "CAD-like" output. However, the design space is more restricted than density-based SIMP and the method is less mature. Concluded that SIMP remains the better starting point for this project due to its wide adoption and compatibility with the skeletal reconstruction approach.

### Research on Ge Yin's Work
Studied Ge Yin's 2019 PhD thesis and 2020 paper in detail. Key insights:
- Homotopic thinning algorithm preserves topology (connectivity, holes, branches)
- Skeleton graph extraction produces nodes (junctions/endpoints) and edges (beam centrelines)
- CSG reconstruction with cylinders at edges and spheres at junctions
- Most complete algorithmic description available, including pseudocode

### Relationships Between Major Methods
Compiled a comparison table of major approaches showing how they build on each other:

| Method | Category | Innovation |
|--------|----------|------------|
| Ge Yin (2019) | Primitive-based | Fits cylinders & spheres locally to skeleton |
| Polak (2023) | Primitive-based (graph-aware) | Adds branch junction blending, hierarchical fitting |
| Amroune (2021) | Lofted surfaces | Lofted B-spline surfaces along skeletons |
| Feng (2025) | Implicit lofting / hybrid | Learned implicit functions to loft between cross-sections |

### Key Insight
"There won't be one system to do it all" — skeletal-based approaches are best for interpretable, manufacturable simple geometries (beams, trusses, structural frames), while NURBS surface-based approaches are better suited for complex organic geometries (aerospace lightweighting, additive manufacturing). This project should focus on the skeletal approach first, with surface extension as a stretch goal.

### Weekly Summary
**Focus:** Literature review, manual workflow exploration, and tool evaluation.
**Work Completed:** Ran Solidworks TO on bracket, set up VS Code and GitHub, categorised techniques into four families, attempted AMRTO code, got Instant Meshes working (3 hours macOS troubleshooting), researched MMC TO and Ge Yin's work, compiled method comparison table.
**Key Achievements:** Hands-on understanding of the manual TO-to-CAD workflow and its pain points. Systematic categorisation of all existing approaches. Deep understanding of Yin's skeleton-based framework as the most complete approach available.
**Reflection:** The manual process in Solidworks confirms the need for automation — even a simple bracket took significant manual effort. The skeleton-based approach looks most promising for producing an *interpretable* output (beams and joints) rather than just a smoothed mesh.

---

## Week 3 — 20–26 October 2025

### Weekly Summary
**Focus:** Deep literature review and method categorisation.
**Work Completed:** Conducted in-depth review of key papers. Explored the two main reconstruction paradigms in detail:

**Cylinder/Sphere Fitting (Yin 2019, Polak 2023):**
- Each skeletal segment treated as a generalised cylinder with varying radius along its length
- Pros: robust for tubular/tree-like structures, easy to parameterise (radius, axis, length), computationally light, handles noisy data well
- Cons: hard to achieve smooth global continuity at junctions, can produce faceted surfaces, not ideal for sheet-like structures
- Best for: pipes, vascular structures, structural trusses — and our target application

**Lofted/Swept Surfaces (Amroune 2021, Feng 2025):**
- Smooth surface interpolated between skeletal cross-sections
- Pros: visually and geometrically smooth, handles non-tubular regions, captures tapering transitions
- Cons: needs well-defined cross-sections, computationally expensive, sensitive to skeleton artifacts
- Best for: anatomical forms, additive manufacturing, aesthetic fidelity

**Hybrid Option (emerging):**
- Fit cylinders to skeleton branches, use radius field interpolation for continuous implicit surface, then mesh or loft the result
- Gives interpretability of cylinder model + continuity of lofted surfaces

Explored FreeTO and Top3DSTL tools in detail. Set up Zotero reference library with 15+ papers categorised by method type.

**Key Achievements:** Clear taxonomy of existing methods. Identified the research gap: no existing tool provides a fully automated pipeline from density field to *parametric, editable* CAD model with structural element recognition (beam vs plate).
**Problems Faced:** Some papers describe methods at a high level without sufficient implementation detail for replication.
**Solutions:** Cross-referenced multiple papers and found Yin's PhD thesis (2019) provides the most complete algorithmic description, including pseudocode for the thinning and graph extraction stages.
**Reflection:** The Yin framework stands out as the most complete and well-documented skeleton-based approach. It will serve as the baseline for my implementation. The hybrid approach (primitives + surfaces) is the long-term direction but too ambitious as a starting point.

---

## Week 4 — 27 October – 2 November 2025

### Supervisor Meeting (30 Oct)

**Questions Prepared:**
- Do I need to justify methodology decisions in the proposal, or just state the intended path?
- What are the trade-offs between parametric primitives (cylinders/spheres) and continuous surface interpolation (lofted/swept surfaces)?
- Can the project focus on building a user-friendly, parameterisable tool that wraps existing methods?
- What makes a successful dissertation project — how much needs to be original?
- Could I create a method which gives recommendations to users about settings based on initial input?

**Discussion Notes:**
- Two broad reconstruction approaches discussed:
  - **(A)** Parametric primitives (cylinders, spheres, cones) — simpler, more relevant to structural/construction applications
  - **(B)** Continuous surface interpolation (lofted/swept surfaces) — more relevant to additive manufacturing and lightweight design
- Decision: Start with skeleton-based cylindrical reconstruction (approach A), then aim to extend to surface-based geometries as a stretch goal
- Discussed project plan structure and Gantt chart requirements
- Supervisor reviewed draft project plan via email beforehand
- Discussed whether project focus should be creating a user-friendly version of existing methods — one input, one output with customisable parameters
- Confirmed: project should aim to first get a working pipeline, then expand

**Actions Agreed:**
- Finalise Updated Project Plan with Gantt chart and milestones
- Begin implementing initial framework before Christmas
- Get a basic pipeline working for a simple beam case by end of term

### Researching Skeletonisation Tools (29 Oct)
Compiled comprehensive list of available skeletonisation tools:

**MATLAB:**
- **bwskel** (Image Processing Toolbox) — Built-in 2D and 3D homotopic skeletonisation for binary images/volumes
- **Skeleton3D** (File Exchange) — Popular 3D thinning (Lee et al. style) for volumes
- **Skel2Graph3D** (File Exchange) — Converts a 3D voxel skeleton into a graph (nodes/edges)
- Typical pipeline: mesh → voxelise → bwskel or Skeleton3D → Skel2Graph3D → prune/simplify

**Python:**
- **scikit-image** — `skeletonize` (2D) and `skeletonize_3d` (3D). Reliable, well-maintained.
- **skan** — Analysis toolkit for skeletons (branch lengths, nodes, graph export)
- **ITK / SimpleITK** — `BinaryThinningImageFilter3D`. Good medical-grade implementation.
- **CGAL (Python bindings)** — Mean-Curvature-Flow skeletonisation directly on triangulated surface meshes
- **VMTK (Vascular Modeling Toolkit)** — End-to-end centerline extraction on surface meshes/volumes

Installed STLREAD in MATLAB and Skel2Graph3D for initial experiments. Also investigated Inpolyhedron for voxelisation.

### Updated Project Plan & Proposal

**Refined Project Objectives:**
1. **Replication:** Reproduce the skeletonisation and CAD reconstruction approaches from Yin (2020), Polak (2023), and Feng (2025)
2. **System Integration:** Combine the most effective components into a single automated pipeline with "one input, one output" functionality
3. **Parameterisation and Usability:** Explore automatic parameterisation of reconstructed geometries for downstream CAD modification
4. **Manufacturability and Constraint Optimisation:** Investigate embedding manufacturing constraints (minimum feature thickness, AM overhangs) into the reconstruction pipeline
5. **Validation:** Test on benchmark TO examples (cantilever beam, bridge, rocker arm)

**Planned Deliverables:**
- Pre-processing code (mesh/voxel input handling)
- Skeletonisation algorithm implementation
- Reconstruction code with CAD package integration
- Refined integrated system with single input/output
- Parametric reconstruction for downstream design modification
- Manufacturability scoring

### Weekly Report
**Focus:** Project planning and literature consolidation.
**Work Completed:**
- Completed and submitted the Updated Project Plan (formative) per ENGI 4093 requirements
- Revised project aims to emphasise user-friendliness, parametric flexibility, and a single integrated program
- Developed detailed Gantt chart with milestones through January
- Completed critical comparison between surface-based, skeleton-based, and hybrid reconstruction approaches
- Reviewed Yin (2019, PhD) and Yin et al. (2020) in detail — identified the homotopic thinning and CSG reconstruction pipeline as the core baseline
- Defined high-level Python module structure: voxelisation, skeletonisation, graph generation, reconstruction
- Drafted and sent professional email to Dr Ullar outlining rationale for chosen framework structure
- Compiled comprehensive list of skeletonisation tools across MATLAB and Python ecosystems

**Key Achievements:** Project direction firmly established with supervisor sign-off. Clear roadmap from literature to implementation.
**Reflection:** The skeleton-based framework is the most promising route, balancing robustness and interpretability. A hybrid extension combining cylindrical primitives with surface lofting was identified as a valuable future direction. This week established a solid theoretical and organisational foundation for transitioning into implementation.

---

## Week 5 — 3–9 November 2025

### Supervisor Meeting (6 Nov)

**Discussion Notes:**
- Supervisor reviewed literature review draft
- Key guidance: "For the lit review — make sure it flows, almost like a story. Don't just list off different papers and what they did."
- Discussed narrative structure: motivation → existing methods → identified gaps → proposed approach

**Actions Agreed:**
- Restructure literature review to follow narrative arc
- Continue refining Initial Project Report sections

### Literature Review Structure Development (5 Nov)
Developed comprehensive literature review structure with reference mapping:

**Section 2.1 — Surface-Based Methods (NURBS Reconstruction):**
Reference mapping:
| Paper | Key Content | Relevance |
|-------|------------|-----------|
| Ren et al. (2022) | Curvature-based segmentation + local B-spline fitting | Algorithmic approach aligning with commercial workflows |
| Park et al. (2020) | B-spline patch fitting guided by principal curvature | Illustrates "semi-automatic" TO-to-CAD |
| Beccari et al. (2025) | Quad meshes before tensor-product B-spline fitting | Modern benchmark, still fails on branching topologies |
| Commercial (Altair, nTopology, TOSCA) | Smoothing and surface fitting | Contrast against academic methods |

Key emphasis: All surface-based approaches focus on *local surface continuity* rather than *topological fidelity* — they create smooth surfaces but cannot preserve complex branching structures.

**Section 2.2 — Skeleton-Based Reconstruction:**
| Paper | Key Content | Relevance |
|-------|------------|-----------|
| Yin & Cirak (2019, 2020) | Homotopic skeletonisation + CSG tree | **Core reference** for project |
| Lee et al. (1994) | 3D digital topology thinning algorithms | Theoretical basis |
| Cuilliere et al. (2021) | Curve skeletons via Laplacian contraction | Comparison — different from voxel thinning |
| Polak et al. (2023) | Voxel reduction + graph simplification | Recent validation of Yin's approach |

Key emphasis: advantage lies in **topology preservation** — maintaining connected components, handles, and holes.

**Section 2.3 — Feature-Based and Hybrid Methods:**
| Paper | Key Content | Relevance |
|-------|------------|-----------|
| Beccari et al. (2025) | Feature primitives + spline interpolation | Hybrid surface-primitive approach |
| Feng et al. (2025) | Implicit radius field interpolation | Hybrid implicit-explicit direction |
| Amroune et al. (2021) | RMF-lofted B-spline surfaces | Bridges skeleton and surface categories |

**Planned Narrative Flow:**
1. Surface-based methods → persistent limitation (poor topology handling) → leads to skeleton-based as solution
2. Skeleton-based → primitive vs surface reconstruction → topology preservation matters
3. Hybrid methods → merge primitives and surfaces → project builds on skeleton but seeks hybrid flexibility

**Section 2.4 — Comparative Summary:**
| Approach | Input | Output | Automation | Topology | CAD Compat. |
|----------|-------|--------|------------|----------|-------------|
| Surface-Based | Mesh | NURBS patches | Semi-auto | Weak | High |
| Skeleton-Based | Voxel/Mesh | Cylinders/Spheres | Full-auto | Strong | Moderate |
| Hybrid | Skeleton + Implicit | Blended surfaces | Moderate | Strong | Good |
| Parametric/MMC | CAD primitives | CAD parametric | Limited | Strong | Excellent |

**Research Gap:** Despite progress, full automation and topological robustness remain unresolved. Existing CAD reconstructions either lose optimality or require manual patching. This project focuses on developing a topologically robust, parametric, user-editable reconstruction framework.

### Initial Pipeline Attempt (3–5 Nov)
Attempted to implement a basic V1 pipeline in Python (VS Code). Proved too troublesome at this stage due to the complexities of different pipeline stages interacting — the thinning, graph extraction, and reconstruction steps each have dependencies that were not yet well understood. Decided to step back and focus on understanding each stage individually before attempting integration.

### Weekly Summary
**Focus:** Report writing, literature review structure, and initial coding attempt.
**Work Completed:** Restructured literature review to flow as a coherent narrative with detailed reference mapping for each section. Developed comparative summary table across all four method families. Attempted first pipeline implementation — revealed gaps in understanding.
**Key Achievements:** Literature review now reads as a coherent argument building towards the project's contribution. Comprehensive reference mapping ensures every claim is backed by specific papers.
**Reflection:** The supervisor's feedback on narrative flow was valuable — the lit review is now much stronger. The failed initial coding attempt was also valuable — it highlighted exactly which algorithmic details I need to understand before implementation can proceed.

---

## Week 6 — 10–16 November 2025

### Weekly Summary
**Focus:** Continued report writing and technical study.
**Work Completed:** Continued drafting the Initial Project Report methodology section. Conducted detailed study of homotopic thinning algorithms, particularly Lee et al. (1994) which is the standard for 3D skeletonisation. The algorithm iteratively removes "simple points" (surface voxels whose removal doesn't change topology) until only the skeleton remains.

Explored Python libraries for implementation:
- **scikit-image**: Morphological operations, distance transforms
- **NetworkX**: Graph construction and analysis
- **Open3D**: 3D point cloud and mesh visualisation
- **NumPy/SciPy**: Core numerical operations, KD-trees for spatial queries

Created initial project repository structure with placeholder modules.
**Key Achievements:** Solid understanding of the thinning algorithm that will form the core of the skeletonisation stage.
**Problems Faced:** The thinning algorithm has multiple modes (curve-preserving, surface-preserving, combined) and choosing the right mode depends on the structural type — this will need careful handling for mixed beam-plate structures.
**Reflection:** The choice of thinning mode is actually a key design decision: mode 0 (curve-preserving) is ideal for beam structures but destroys plate-like surfaces, while mode 1 (surface-preserving) retains sheets but may not capture thin beams properly. This hints at the need for a classification step before thinning — classify regions as beam or plate first, then thin each appropriately.

---

## Week 7 — 17–23 November 2025

### Weekly Summary
**Focus:** Report writing and SIMP method study.
**Work Completed:** Continued Initial Project Report — focused on the background theory section covering the SIMP (Solid Isotropic Material with Penalisation) method. SIMP is the TO method that generates the input density fields for the pipeline. Understanding it thoroughly is essential because:
- The penalty parameter (p) determines density distribution sharpness — higher p gives cleaner 0/1 densities but may lose thin structural features
- The volume fraction constraint directly affects the skeleton topology
- The filter radius controls minimum feature size

Studied the MATLAB Top3D code (Liu & Tovar, 2014) which will serve as the reference TO solver for generating test inputs. Began planning the Python module structure for the baseline pipeline: `run_pipeline.py` → `topology.py` → `thinning.py` → `graph.py` → `reconstruct.py`.
**Key Achievements:** Complete understanding of the SIMP method and how its parameters influence the downstream skeleton extraction.
**Reflection:** There's a clear dependency chain: TO parameters → density field quality → skeleton quality → CAD output quality. This means the pipeline needs to be robust to a range of TO parameter settings, not just tuned for one specific case.

---

## Week 8 — 24–30 November 2025

### Supervisor Meeting

**Topics Discussed:**
- 2D Skeletonisation → how it works → how 3D was made from 2D
- Change of input to .mat binary file → using voxel input instead of converting to mesh STL
- Trying surfaces/plates alongside beams
- Trying new inputs of variable difficulty — some more suited to cylinders/spheres, also trying more complex structures similar to those in Yin's papers
- Key discussion: **defining the goal of the reconstruction program:**
  - Is it to reconstruct the mesh as close to the original output as possible?
  - Is it to reconstruct the TO mesh to have the same geometric properties and/or structural properties and integrity?
- Need to work on fusing as it can mess some imports up

**Working On:**
- Getting more inputs of variable difficulties, different radii, more and less tubular
- Definitely focusing the project towards reconstruction using tubular members at present
- Working on a way to view the skeleton before CAD — will allow for quicker debugging
- Once skeletonisation is working well, would like to expand towards using cross-sections — but quite far off
- How to open and view voxel files; different inputs other than .vox

**Actions Agreed:**
- Still need to work on radii detection from cross sections — where in the process to take the cross section from and how this correlates with the sectioning of the chain/members

### 3D Skeletonisation Pipeline Refinement
Confirmed Python pipeline correctness — output files generating correctly:
- `frame3d_nodes.csv` — skeleton node positions with radii
- `frame3d_edges.csv` — connections between nodes
- `frame3d_plates.stl` — extracted plate geometry as STL mesh

Identified that the issue was on the FreeCAD side, not in the Python skeleton extraction. Investigated the difference between outputting geometry as surface shells versus solid volumes — shells are lighter but cannot be used for FEA without thickening.

### FreeCAD Macro Development — Iterative Fixes

**Initial macro behaviour:**
- Import STL via ImportGui → create shape from mesh → convert to solid → fuse rods + plates
- Observed repeated errors: unsupported format issues, mesh-to-solid failures, partial solids, catastrophic boolean fusions (warped, stretched faces)

**Fixed STL import pathway:**
- Rewrote the import pipeline using `Mesh.Mesh` rather than `ImportGui`
- Ensured mesh topology passed correctly into `Part.Shape.makeShapeFromMesh()`
- Removed problematic fuse logic

**Finalised "rods + plates (no fuse)" macro:**
- Loads rods (spheres & cylinders) as a compound
- Imports plate STL as a Part shape (PlateShape)
- Attempts solid conversion (PlateSolid) **but never fuses**
- Result: reliable rod geometry, visually correct plates as surfaces, no boolean distortion

**Verified plates appear as surfaces** — confirmed that plates appear surface-only because STL → Part shape produces a shell, not a thickened volume. Plate solids were only partially generated due to incomplete shells or non-manifold edges.

### CAD Strategy Shift — FreeCAD "Thickness" Tool
- Confirmed that FreeCAD refuses to generate solids via `makeSolid()` unless all STL shells are perfectly closed
- Identified that marching-cubes output can introduce small topological inconsistencies
- Determined that Part → Thickness:
  - Reliably converts surface shells to solid volumes
  - Allows explicit control over plate thickness
  - Bypasses need for perfect manifold topology
  - Avoids boolean operations entirely
- Planned macro extension to automatically apply Thickness after importing PlateShape

### 2D Skeletonisation — Robust Framework & Variable Radius Pipeline (24 Nov)
- Tested mesh → slice → raster workflow; confirmed slicing with trimesh works but `to_planar()` throws deprecation warnings
- Built working 2D skeletonisation script using `skimage.skeletonize`. Extracted nodes/edges and exported CSV for FreeCAD
- Added DXF export of polylines and PNG overlay of mask + skeleton
- Added medial-axis distance transform for radius estimation. Hit error with `sampling=` in medial_axis; switched to `scipy.ndimage.distance_transform_edt`
- Added per-node and per-edge radii to CSV output. Updated FreeCAD macro to read radius values
- First tests: cylinder radius too noisy due to per-pixel medial-axis noise
- Added **Option 1: constant radius per chain** (branch radius = median of raw radii)
- Implemented chain extraction from graph (junctions/endpoints start chains)
- Added **global regularisation** of chain radii (compress to ~0.7×median → 1.3×median)
- Added **branch pruning**: skip chains shorter than relative threshold (0.02 × bounding box)
- Tuned RDP simplification alongside pruning — branches now much cleaner
- FreeCAD output generating cleaner, more uniform frames

### Adding MagicaVoxel .vox Input (25–26 Nov)
Extended pipeline to accept MagicaVoxel .vox files exported from MATLAB Top3D code:

**MATLAB side:**
- Reviewed `write_magica_vox` implementation: standard MagicaVoxel v150 format
- Header: "VOX " + version 150, then SIZE chunk (Nx, Ny, Nz), XYZI chunk (N voxels with x,y,z,colorIndex), RGBA palette chunk
- Important: .vox does not store physical voxel size (pitch) or origin — these must be passed via command line

**Python side — new pipeline:**
Created `voxel_to_3d_frame_variable_radii_vox.py` with:
- MagicaVoxel loader function
- CLI interface (`-pitch`, `-origin`, `-rdp_eps`, `-min_branch_len`, `-out`)
- Reused core pipeline: `compute_skeleton_3d` → `build_skeleton_graph` → `rdp_indices_3d` → `decompose_into_chains` → `simplify_graph_rdp_3d` → `prune_small_branches` → `compute_radii_from_edt` → `save_frame_csv`

### General Debugging Notes
- Diagnosed extreme sliver surfaces and warped geometry during boolean fusion
- Concluded: plate solids were incomplete, fusion attempted with non-manifold shells, FreeCAD boolean kernel (OpenCascade) generated invalid geometry
- Resolved by removing fusion: rods as solids, plates as surface shapes (or thickened solids)

### Decision on Project Goal
After discussion with supervisor, defined the reconstruction goal as: **"To create editable geometry which accurately represents the outputted TO mesh's structural properties."** This is more meaningful than pixel-perfect mesh reproduction — the CAD model should be structurally equivalent but parameterically editable.

### Weekly Summary
**Focus:** Initial Project Report finalisation, pipeline debugging, and .vox input support.
**Work Completed:** Confirmed Python pipeline correctness, developed FreeCAD macro through multiple iterations, identified FreeCAD "Thickness" as preferred method for plate solids, built 2D skeletonisation with variable radii, added MagicaVoxel .vox input support, resolved boolean fusion issues.
**Key Achievements:** Pipeline now accepts .vox input files. FreeCAD macro stabilised by removing boolean fusion. Clear reconstruction goal defined.
**Problems Faced:** Boolean fusion of rods and plates caused catastrophic geometry distortion in FreeCAD. Per-pixel radius estimation was too noisy.
**Solutions:** Abandoned boolean fusion — keep rods and plates as separate clean geometry. Added chain-level radius averaging and global regularisation.
**Reflection:** Writing the methodology forced me to think through each stage in detail. The question of "reconstruct geometry vs preserve structural properties" is fundamental — my pipeline needs to support both, but structural fidelity is the primary goal.

---

## Week 9 — 1–7 December 2025

### Supervisor Meeting

**Discussion Notes:**
- Brief meeting focused on report submission timeline
- Discussed plan for Christmas break coding sprint
- Supervisor mentioned: could look into feature recognition / preserving geometric constraints and connections (specified holes, etc.)

### Chain Simplification and Visualisation Verification
- Increased RDP epsilon to 2.0 for merging collinear nodes — significantly reduced node count while maintaining topology
- Fixed visualisation `ImportError` (deprecated `skeletonize_3d` function in scikit-image)
- Before simplification: high density of nodes along straight sections
- After simplification (rdp_eps=2.0): clean, straight chains with nodes only at corners and junctions

### Weekly Summary
**Focus:** Report finalisation and pipeline refinement.
**Work Completed:** Final proofreading and formatting of Initial Project Report. Formatted bibliography consistently. Created appendix with preliminary test case descriptions. Refined graph simplification — RDP with increased epsilon merges collinear nodes effectively.
**Key Achievements:** Initial Project Report submitted. Graph simplification working well.
**Reflection:** The report-writing phase was invaluable for crystallising methodology. I now have a clear roadmap for the coding phase.

---

## Week 10 — 8–14 December 2025

### Supervisor Meeting

**Questions Prepared:**
- How much investigation vs reporting of what works best? How much detail for approaches that weren't ideal?
- Should I investigate which method is best for each pipeline stage, or find the best one and optimise it?

**Discussion Notes:**
- Supervisor advised: "Find the best that works, then optimise that" — don't spend too much time on exhaustive comparisons
- Discussed test case strategy: range of difficulties from simple beam to complex hybrid structures

**Actions Agreed:**
- Set up GitHub repository properly
- Create test cases of ranging difficulty
- Test methods on test cases using FEA to compare initial vs reconstructed geometry
- Submit Initial Project Report
- Submit Updated Project Plan (5 pages)

### Weekly Summary
**Focus:** Setting up for implementation phase.
**Work Completed:** Set up GitHub repository with proper structure. Submitted Updated Project Plan. Planned test case suite:
1. Simple cantilever beam (baseline)
2. Bridge structure (multiple beams, joints)
3. Plate-like structure (test surface handling)
4. Hybrid beam-plate (the hard case)

Began preliminary coding — setting up Python package structure.
**Key Achievements:** Clean GitHub setup. Clear test case plan with increasing difficulty.
**Reflection:** The "find best then optimise" guidance is practical — need to avoid analysis paralysis.

---

## Christmas Break — 15 December 2025 – 26 January 2026

### Development Log

**3 December — Fixing Surface Skeleton Issue:**
- Skeletonisation of hollow voxel models produced a "surface skeleton" (mesh-like structure on beam surfaces) rather than a centreline skeleton
- Root cause: medial axis transform reduces a hollow shell to a medial surface, whereas it reduces a solid volume to a centreline
- Solution: implemented hole-filling step using `scipy.ndimage.binary_fill_holes` before skeletonisation
- Created V2 script (`voxel_to_3d_frame_variable_radii_vox_V2.py`) with `-no-fill` flag to optionally disable

**3 December — Skeleton Visualisation Tool:**
- Added `-visualize` flag using Open3D integration
- Visual elements: skeleton voxels (grey PointCloud), graph branches (coloured LineSets per chain), junctions (red spheres at degree ≠ 2 nodes)
- Verified script syntax and argument parsing; Open3D window successfully displays expected geometry

**27 December — Variable Radius Pipeline:**
- First working code: basic voxel-to-beam conversion with variable radii based on EDT
- Core insight: EDT values at skeleton points approximate local "thickness" — gives natural beam radii

**29 December — Code Reorganisation:**
Restructured entire project from monolithic script into a modular Python package:
```
src/
├── io/          (voxel_loader.py, exporter.py)
├── processing/  (skeletonization.py, graph.py, simplification.py, pruning.py, radius.py)
├── visualization/ (visualizer.py)
main.py
requirements.txt
```
Each function moved to its logical module. Verification: run refactored `main.py` with `-help` flag.

**31 December:**
- Implemented advanced plate detection with contour extraction and FreeCAD integration
- Restored paper replication features: RMF sweeps, cubic Bezier curves, FreeCAD macro
- End-of-year milestone: basic pipeline runs end-to-end for simple beam structures

### Loft Implementation Plan
Created detailed implementation plan for extending pipeline to use FreeCAD loft surfaces:
1. **Cross-Section Extraction**: At each node, define a cutting plane (normal = chain tangent), interpolate the 3D voxel grid on the plane, use `skimage.measure.find_contours` for boundary polygon
2. **FreeCAD Macro Generation**: Python macro that iterates through edges, creates sketches for each cross-section, creates Loft between them
3. **Refactoring**: Isolate FreeCAD export logic, add `-loft` flag

### Summary & Reflection
The Christmas break was the critical transition from research to implementation. Key decisions:
- **EDT for radius estimation** works well and is parameter-free
- **FreeCAD** chosen over Solidworks API (open-source, scriptable, no licensing)
- **Plate detection** was attempted but proved unreliable with initial threshold approach

The codebase proved the concept works but highlighted two major challenges: (1) robust plate vs beam classification, and (2) FreeCAD macro stability.

---

## Week 13 — 27 January – 2 February 2026

### Supervisor Meeting

**Discussion Notes:**
- Limited project work this week due to AI coursework deadline
- Discussed topology optimisation techniques and which TO solver to use
- Compared MATLAB Top3D (well-documented, established) vs Python alternatives

**Actions Agreed:**
- Complete AI coursework
- Plan the term ahead with specific weekly objectives
- Get cantilever beam working through full pipeline with images for report

### Rebuilding Skeletonisation from Scratch (2 Feb)
Previously using Skeletonise3D external package which worked but could not be modified. Rebuilt own skeletonise function for full control and debuggability.

Key terminology for dissertation:
- **Homotopic Thinning** = getting rid of one voxel at a time, checking neighbourhood patterns to determine if removal is safe
- Planned diagram for report: different neighbours and whether to remove or not

### Switch to Competition Classification (2 Feb)
Identification of plate vs rod/branch is difficult and can be done in many ways — this became a core research focus.

Previous approach relied on a single threshold for Linearity (C_l), which misclassified structures or failed to capture full geometry.

**New approach — Competition Classification:**
Voxels classified based on the **maximum** of Linearity (C_l), Planarity (C_p), and Sphericity (C_s) from Westin metrics:
- **High Linearity** (λ₁ >> λ₂) → Plate surface
- **High Planarity** (λ₁ ≈ λ₂ >> λ₃) → Rod surface

Impact on parameters:
- `plate_thresh` → **now ignored** (code automatically compares Linearity vs Planarity)
- `protect_thresh` → **still very important** (controls angle of attack for erosion defence)
  - Higher (e.g., 0.9): Very strict — locks entire plate surface
  - Lower (e.g., 0.5): Looser — allows some erosion on curved parts
  - Recommended: ~0.8–0.9

### `plate_thresh` Parameter Analysis
Detailed analysis of how the threshold gate interacts with Westin metrics classification:
- **Low threshold (0.3):** "Loose" filter — base plate correct BUT rods also turned blue, rods fail to skeletonise
- **High threshold (0.9):** "Strict" filter — rods correctly rejected BUT plate may have holes
- **Optimal (0.6–0.8):** "Sweet spot" — base plate protected, rods skeletonise correctly

### Testing Plate Preservation on Multiple Geometries (2 Feb)

**Objective:** Address critical failure in homotopic thinning where plate-like structures were incorrectly eroded into 1D lines. Establish a **Hybrid Skeleton** preserving plates as 2D sheets while thinning rods to 1D centrelines.

**Challenges and Solutions:**

**1. The "Disappearing Plate" Issue:**
- Standard algorithm eroded base plate of bridge model into sparse grid of lines
- Standard algorithm prioritises endpoint preservation (suitable for rods) but fails to respect surface topology
- Solution required: distinguish "Plate Voxels" from "Rod Voxels" with distinct thinning rules

**2. Robust Classification (Westin Metrics):**
- Implemented Structure Tensor Analysis (Linearity C_l, Planarity C_p)
- Issue: classification brittle — plate edges naturally exhibit high curvature → misclassification
- Fix: **"Deep Capture" strategy** — Competition + Dilation + Interior Propagation

**3. Implementing Yin's Topological Rules:**
- New routine (`skeletonization_yin.py`) evaluating 2×2×2 neighbourhoods
- **The Rotation Problem:** Initial implementation failed to preserve tilted plates
- Fix: programmatically generate all **24 rotational symmetries** of the surface templates

**4. Passive Region Preservation:**
- `-solid_plates` flag to bypass thinning for classified plate voxels

**Test Results — 4 Geometries at Multiple Thresholds:**

| Geometry | thresh=0.8 | thresh=0.9 | thresh=0.95 | thresh=0.999 |
|----------|------------|------------|-------------|---------------|
| Synthara Quadcopter Frame | Good plate capture | Some plate erosion | Conservative | Strict — only clear plates |
| Simple Bridge (Fusion 360) | Full base plate | Base plate preserved | Minor holes | Partial plate |
| Rocker Arm | Good separation | Clean beam/plate split | Conservative | Mostly beams |
| Bridge from Solidworks TO | Full plate, clean beams | Good balance | Some plate erosion | Beams only |

**Conclusion:** Pipeline rendered robust. Combining gradient-based classification with edge dilation and topological rules (Yin's surface points) reliably produces hybrid skeletons. Next stage: extend to the reconstruction macro.

### Plan for Term Ahead
- Week 13–14: Working pipeline with plate detection
- Week 15–16: Optimisation stages (size and layout)
- Week 17–18: Full test case results and report writing
- Week 19–20: Report finalisation and submission

### Weekly Summary
**Focus:** Coursework completion, term planning, and plate detection implementation.
**Work Completed:** Submitted AI coursework, rebuilt skeletonisation from scratch, switched to Competition Classification, implemented Deep Capture strategy, tested on 4 geometries with multiple thresholds, generated all 24 rotational symmetries for surface patterns.
**Key Achievement:** First working plate detection tested across multiple geometries with robust Westin metrics classification.
**Reflection:** Having a structured term plan is essential — only 8 weeks until submission. The plate detection is encouraging but a single threshold approach won't work for all geometries — need multi-signal classification.

---

## Week 14 — 3–9 February 2026

### Supervisor Meeting (3 Feb)

**Discussion Notes:**
- Reviewed term plan and pipeline priorities
- Discussed strategy for integrating TO solver with reconstruction pipeline

### Research Gap Identification (4 Feb)
Conducted systematic analysis of Yin/Polak pipeline limitations:
- Plate-like regions collapsed into rods during thinning
- Flat surfaces lost — only 1D skeleton preserved
- Hybrid structures (beams + plates) not handled
- Reframed dissertation goal: *"hybrid skeleton-based CAD reconstruction framework that preserves both frame-like and surface-like load paths"*
- Core idea: detect rods and surfaces DURING skeletonisation, not after
- Planned extension: typed graph (curve graph for beams, surface graph for plates, interface edges)

### Baseline Framework Planning (5 Feb)
Created comprehensive implementation plan for reproducing Yin paper as baseline:

**Directory structure:** `src/pipelines/baseline_yin/`

**Methodology — Algorithm Implementation Plan:**
1. **Algorithm 3.1 — Thinning:** Homotopic thinning with simple point detection, iterative layer-by-layer removal
2. **Algorithms 4.1 & 4.2 — Graph Construction:** Convert skeleton voxels to graph with nodes and edges, chain decomposition from skeleton connectivity
3. **Algorithms 4.3–4.5 — Graph Cleanup:** Prune short branches (spurious noise), merge close nodes (collapse threshold), simplify with RDP (Ramer-Douglas-Peucker)

**Pipeline Architecture (4 Stages):**
- Stage 0: SIMP Topology Optimisation (input generation)
- Stage 1: Skeleton Reconstruction (thinning → graph → cleanup → radius estimation)
- Stage 2: Layout Optimisation (node position refinement via FEA)
- Stage 3: Size Optimisation (radius/cross-section optimisation via FEA)
- Output: JSON with nodes, edges, radii → FreeCAD macro

### Radius Estimation Modes
Added `--radius_mode` flag with two strategies:

1. **EDT mode (default):** "Reverse Engineering" — uses Euclidean Distance Transform values at skeleton points to estimate local thickness. Direct geometric reconstruction capturing variable thickness from the already-optimised geometry.

2. **Uniform mode:** Yin's Volume Matching rule — physics-based initialisation ensuring starting mass equals TO voxel mass. Formula: $A_0 = V_{total} / L_{total}$, $r_0 = \sqrt{A_0/\pi}$. All beams same initial radius.

**For dissertation (if skipping Size Optimisation):** Use EDT mode for direct geometric reconstruction.

### Extending Size Optimisation to Hybrid Structures
Documented the mathematical framework for hybrid rod+plate optimisation:

**The Challenge:** Yin's baseline only optimises rod radii ($r_i$). How to incorporate plates?

**The Solution — Generalised Size Optimisation:**
- **Variables:** Rods have cross-sectional area $A_{rod}$ (radius $r$), Plates have thickness $t_{plate}$
- **Volume constraint:** $V_{total} = \sum(A_{rod} \cdot L_{rod}) + \sum(t_{plate} \cdot Area_{plate})$
- **FEA:** Mix of Beam Elements and Shell Elements
- **Sensitivities:** Optimizer calculates $\partial C / \partial r_i$ for rods and $\partial C / \partial t_j$ for shells
- **Result:** Material "flows" between rods and plates — critical plates get thicker, useless rods get thinner

**CAD Export Implication:** Pipeline exports both curve data (start, end, radius) and surface data (mesh vertices, thickness).

### Major Development Sprint

**2 February:**
- Implemented Hybrid Medial Surface Thinning V2 with solid plate preservation

**5 February:**
- Completed Hybrid Reconstruction Pipeline V1: end-to-end from TO output to FreeCAD model
- Robust visualisation at each pipeline stage
- Graph cleanup: spurious branch removal, close node merging
- Stable FreeCAD macro with graceful error handling

**8 February:**
- Added Synthara Frame test problem with hub-and-spoke BCs
- Created unified optimisation pipeline with orchestration
- Implemented radius-aware bounding box constraints

**9 February:**
- Fixed cantilever beam BCs: base fixed (min-coord), tip loaded (max-coord)
- Distributed load across all tip nodes to handle forked skeleton topologies

### Weekly Summary
**Key Achievements:** First complete working pipeline running multiple test cases. Hybrid thinning correctly distinguishes beam (1D) and plate (2D) elements. Comprehensive baseline framework planned and partially implemented.
**Problems Faced:** Initial BC formulation was inverted. Forked skeletons caused load concentration.
**Solutions:** Fixed coordinate mapping. Distributed loading across all nodes at loaded face.
**Reflection:** Most productive week so far. Pipeline running end-to-end is a major milestone.

---

## Week 15 — 10–16 February 2026

### Supervisor Meeting

**Discussion Notes:**
- Demonstrated working pipeline on cantilever beam and bridge test cases
- Discussed zone classification approach for automatic beam/plate distinction
- Topic: Fix Top3D integration into VS Code pipeline

### Generating Hybrid Beam-Plate Test Inputs
Documented three recommended configurations that produce hybrid topologies containing both beam-like and plate-like features:

**1. The Torsional Box (Strong Hybrid):**
Load at a corner of a thick domain → creates "skin" (plate) for torsion resistance and internal "bracing" (beams) for bending.
```
--nelx 60 --nely 30 --nelz 30 --volfrac 0.15 --rmin 3.0
```
*Offset load forces shell-like outer surfaces for torsional rigidity.*

**2. The Multi-Axis Cantilever:**
Diagonal load → complex 3D junctions that flatten into plates.
```
--nelx 80 --nely 40 --nelz 20 --volfrac 0.2 --rmin 2.5 --load_fy -1.0 --load_fz 0.5
```
*Combined Y+Z loads prevent simple 2D frame resolution.*

**3. The "Shell-Reinforced" Frame:**
Large `rmin` relative to domain height → thin members merge into plate-like webs.
```
--nelx 100 --nely 40 --nelz 10 --volfrac 0.25 --rmin 4.0
```

**Key insight:** Previous inputs with `--nelz 4` are essentially 2D slices that almost always result in pure beams. Need `nelz ≥ 20` for significant plate formation.

### Surface Skeletonisation Pipeline Issues
- **Issue identified:** Long chains being detected as plates was due to the long chains being curved
- Idea: split the skeleton at topological junctions then classify each segment
- Need to investigate how beams can branch off one another
- Considered B-spline vs Cube/Cuboid reconstruction in FreeCAD

### Major Development

**10 February:**
- Cleaned up pipeline code, created production-quality version
- Ran 150×50×4 cantilever beam through full pipeline — first publication-quality result
- Implemented BC tag propagation using proximity-based spatial matching

**11 February:**
- Added parameter tuning module (Stage 1 variables) using geometric similarity objective
- Switched from MATLAB Top3D to Python-native PyTopo3D (no MATLAB dependency)
- Created beam-surface input creation tools for custom test geometries

**13 February:**
- Intensive FreeCAD macro debugging — created 6 variant macros to isolate crash sources
- Root cause: `shell.removeSplitter()` and `shell.makeOffsetShape()` are unstable
- Implemented hybrid beam-plate optimisation with volumetric plate extraction
- Refined zone classifier: multi-signal scoring using PCA shape, aspect ratio, BC tag density, EDT uniformity

**15–16 February:**
- Fixed zone classification semantics (zone 1 = plates/cyan, zone 2 = beams/red)
- Full working hybrid pipeline with surface reconstruction via marching cubes
- Initial implementation of Yin-based surface skeletonisation

### Weekly Summary
**Key Achievements:**
- Zone Classification V2 with 4-signal weighted scoring — significant improvement over single thresholds
- FreeCAD macro stabilised using graceful degradation
- 150×50×4 cantilever running cleanly through all stages

**Problems Faced:**
- FreeCAD crashes with complex CSG operations
- Single PCA threshold unreliable for ambiguous shapes

**Solutions:**
- FreeCAD: Direct `Part.Face()` creation from wire polygons. `Part.Solid(shell)` for closed meshes, shell as fallback.
- Zone classification: Multi-signal scoring system:
  1. PCA Shape (weight 0.3): planarity and linearity from eigenvalue ratios
  2. Aspect Ratio (weight 0.3): eigenvalue ratio — plates ≈ 1.0, beams << 1.0
  3. BC Tag Density (weight 0.2): load regions → plate, support regions → beam
  4. EDT Uniformity (weight 0.2): coefficient of variation — plates uniform, beams taper

**Reflection:** Multi-signal classification was a key insight. No single metric works for all geometries — combining four independent signals gives robust classification across all test cases.

---

## Week 16 — 17–23 February 2026

### Plan for the Week
- Sphinx documentation setup for codebase
- Sort logbook (make it consistent, dated, searchable)
- Decide on report structure for BC representation
- bc_tags vs extrema anchoring comparison for boundary condition handling

### Exploring Research Contributions
Documented potential novel contributions of the project:

**Stress-Informed Skeleton Refinement — The Dissertation Angle:**
Current Stage 1 (skeleton extraction) uses purely geometric heuristics (`prune_len`, `collapse_thresh`). These are blind filters that might:
- Prune a short but critical load path because it looks like "noise" geometrically
- Keep a long but useless branch just because it's long
- Merge nodes that should be separate to handle distinct load paths

By the time Stage 2/3 run, the topology is "locked in" — layout/size opt can only improve *that specific graph*.

**Proposed approach:** Use stress field from Stage 0 to make smarter decisions during graph generation:
- "This branch is short but has high Von Mises stress → **KEEP IT**"
- "This branch is long but has zero stress → **PRUNE IT**"

This ensures the starting topology for Stages 2 & 3 is structurally superior. **Novelty:** First method to directly couple topology optimisation stress fields with skeleton refinement heuristics.

### Parameter / Feature Matrix
Created comprehensive comparison across codebase branches:

| Feature | Current Branch | Hybrid Integrated | Curved Beams V2 |
|---------|---------------|-------------------|-----------------|
| Top3D (KDTree, CG solver, passive void) | Yes (latest) | No (old) | No (old) |
| Beam-only reconstruction | Yes | Yes | Yes |
| Iterative size+layout opt | **Yes** | No | No |
| Zone classifier (beam vs plate) | No | Yes | **Yes (latest)** |
| Plate extraction + mid-surface | No | Yes | **Yes** |
| Joint creation (beam↔plate) | No | Yes | **Yes** |
| Curved beams (Bezier) | No | No | **Yes** |
| Coupled beam+shell FEM | No | No | **Yes** |
| Shell element (CST) | No | No | **Yes** |
| FreeCAD: beams only | Yes | Yes | Yes |
| FreeCAD: plates + joints | No | Yes | **Yes** |
| FreeCAD: curved beams | No | No | **Yes** |

### Pipeline Audit
Conducted comprehensive audit documenting every parameter, known failure modes, mitigations, processing times at different resolutions, and which parameters affect which outputs.

### Major Development

**17 February:**
- Set up Sphinx documentation infrastructure with clean configuration
- Added B-spline fit surfaces to plate reconstructions in FreeCAD macro
- **Critical fix:** Corrected size→layout optimisation order (was layout→size, causing compliance spikes)
- Iterative optimisation now converges after 2 iterations — important result for dissertation
- Added `.gitignore`, created dissertation outline and notes folder

**19 February:**
- Implemented curved beam representation using cubic Bezier curves
- Added endpoint spheres to curved beam sweeps to close joint gaps
- Improved arc-length parameterisation and tighter bulge constraints

**20 February:**
- Exported every size/layout iteration as named stages in history JSON
- Multiple FreeCAD macro iterations for stability
- Switched to straight-line beam rendering throughout (curved beams for visualisation only)

**21 February:**
- Fixed outlier node removal before extrema fixing
- Corrected coordinate system mapping (world-X = nelx length, world-Y = nely height)
- Fixed load direction propagation through entire pipeline to FEM

### Weekly Summary
**Key Achievements:**
- Sphinx documentation: 46 pages, 0 warnings, clean architecture diagram
- Curved beam Bezier fitting working for smooth visualisation
- Optimisation order fix resolved compliance spike — converges reliably
- Comprehensive pipeline audit documented all parameters and failure modes

**Problems Faced:**
- Curved beam FEM unreliable: short beams (1–5 voxels) → ill-conditioned stiffness matrices → NaN compliance
- FreeCAD macro crashing with curved beam + plate combinations

**Solutions:**
- Curved beams for geometry/visualisation only — FEM always uses straight beams, Bezier re-fitted after optimisation
- FreeCAD: ball-and-stick macro for reliable rendering

**Reflection:** Curved beam FEM failure was disappointing but the pragmatic geometry-only decision was correct. The optimisation objective is cross-sections and node positions, not curvature.

---

## Week 17 — 24 February – 2 March 2026

### Plan for the Week
- 150×50×4 cantilever beam: full results and write-up
- Complete Sphinx documentation
- Ensure notation consistency across report

### Major Development

**22 February:**
- Implemented density-gradient voxel colouring using RdYlGn colourmap:
  - Red (t≈0): voxels just above vol_thresh (0.3) — "barely surviving" material
  - Yellow (t≈0.5): medium density
  - Green (t≈1.0): fully dense voxels
- Comprehensive documentation overhaul: module docstrings, `__all__` exports, graphviz SVG architecture diagram
- Created `requirements.txt` for reproducibility
- Experimented with Heaviside β-projection in SIMP — reverted (destabilising)
- Added post-optimisation dead-weight pruning and safeguarded hybrid layout opt
- Added `roof_slab` problem type (thin slab with interior point supports)

**23 February:**
- Created LaTeX dissertation outline (`DISSERTATION_OUTLINE.tex`)
- Added LaTeX parameters reference document

**24 February:**
- Added SVG export to convergence figures for editability
- Implemented `--mesh_input` CLI flag for external TO solver meshes
- Added quadcopter problem with branching arm topology
- **Material stiffness consistency fix:** Changed E₀ from 1.0 to 1000.0 in top3d.py to match frame FEM (E=1000). SIMP is scale-invariant in E (optimal density distribution unchanged) but compliance values scale as C_new = C_old / 1000. Effect: SIMP final ≈ 2,236 (was 2,235,716 with E=1), Frame initial ≈ 1,326 — now same order of magnitude for meaningful dissertation comparison.

### Documentation Standards Completed
- Sphinx build: 0 warnings, 46 pages, clean pipeline diagram
- `reconstruct.py`: 25-line module docstring (entry points + pipeline position)
- `top3d.py`: Full `Top3D.optimize(max_loop, tolx)` Args/Returns docstring
- `requirements.txt`: Created at repo root
- `__all__` exports: Added to 7 `__init__.py` files
- Graphviz SVG: Installed graphviz 14.1.2 via homebrew
- Architecture diagram: `.. graphviz::` digraph with hybrid paths dashed, 7 pipeline stages

### Supervisor Meeting (3 Mar)

**Discussion Notes:**
- Reviewed convergence plots and optimisation results
- Volume fraction notation — must be consistent (v_f throughout)
- Material stiffness consistency between continuum and frame FEM

**Actions Agreed:**
- Consistent notation in all figures
- Complete results set for cantilever at multiple resolutions
- Begin writing results chapter

### Notes on Input Variables
Variables are mostly geometry-dependent → can make recommended parameters for different problem types (e.g., recommended settings for cantilever beams, bridges, etc.).

### Weekly Summary
**Key Achievements:**
- E₀ consistency fix enables meaningful SIMP vs frame compliance comparison
- Density-gradient visualisation shows material distribution clearly
- Quadcopter problem demonstrates pipeline handles branching topologies
- Comprehensive Sphinx documentation complete

**Problems Faced:**
- Heaviside projection destabilised SIMP convergence
- Quadcopter passive void placement: centering at arm tips surrounded fixed nodes with E_min → CG diverged

**Solutions:**
- Reverted Heaviside, kept p-continuation
- Centre passive voids at corner positions. Column BCs (full z-columns) force in-plane arm formation

**Reflection:** The E₀ consistency issue could have made the report's compliance comparison meaningless. Setting E₀=1000 in both makes the progression from continuum to frame model numerically coherent.

---

## Week 18 — 3–9 March 2026

### Supervisor Meeting

**Questions Prepared:**
- Should I include implementation details? Or save for documentation/appendix?
- How many appendix pages are we allowed?

**Discussion Notes:**
- Look at TexText plug-in for Inkscape (LaTeX equations in SVG figures)
- Make sure volume fraction is consistent across report
- Include graph for geometric reconstruction case
- Text in diagrams may be too small
- Make sure size and layout optimisation convergence tolerance and constraints are clearly stated
- Viva dates: 5th/6th/7th

**Actions Agreed:**
- Get FreeCAD rendered results for report figures
- Focus on writing methodology chapter
- Run complete test case suite

### Two-Pass Thinning Classification Algorithm
Refined the classification method that replaces PCA-based post-thinning classification:
- **Pass 1:** mode=3 skeleton (surface + curve preserving) — retains 2D sheets and 1D curves
- **Pass 2:** mode=0 skeleton (curve-only) — collapses surfaces to medial curves
- **Difference** (mode=3 present, mode=0 absent) = **plate interior voxels**

This is a purely topological test — zero thresholds, zero PCA, no parameter tuning. Uses connected components + size filter + global linearity filter for noise rejection.

### Weekly Summary
**Focus:** Report writing and algorithm refinement.
**Work Completed:** Wrote methodology chapter sections. Refined two-pass thinning classification.
**Key Achievements:** Two-pass thinning classification is a significant methodological contribution — parameter-free topological approach.
**Reflection:** The two-pass approach is elegant because it exploits the mathematical difference between two well-defined thinning modes. Instead of classifying voxels directly (requiring threshold tuning), it asks: "does this region survive when surfaces are preserved but not when only curves are preserved?" If yes, it's plate-like.

---

## Week 19 — 10–16 March 2026

### Supervisor Meeting (13 Mar)

**Questions Prepared:**
- How should hybrid beam-plate extension be positioned in report?
- Which test case results are most important?
- What constitutes the project's "original contribution"?

**Discussion Notes:**
- Supervisor agreed the two-pass classification is a novel contribution
- Reviewed list of possible contributions:
  1. Multi-signal zone classification (V2 scoring)
  2. Two-pass topological beam/plate classification
  3. Hybrid beam-plate reconstruction pipeline
  4. Curved surface preservation via relaxed surface point detection

**Actions Agreed:**
- Complete results for all test cases with before/after comparisons
- Write up contributions clearly
- Prepare final results figures

### Technical Work: Curved Surface Investigation

**Root Cause Identified:** The `is_surface_point()` function checks whether ALL 8 octants in a 3×3×3 neighbourhood match one of 12 flat plane patterns. On voxel grids, **curved surfaces create staircase patterns** where 1–3 octants at step boundaries fail the strict plane test → mode=3 thinning deletes these voxels → curved surfaces collapse to 1D curves (identical to mode=0) → two-pass difference ≈ 0.

**Proposed Fix:** `is_surface_point_relaxed(hood, min_octants=6)` — requires ≥6/8 octants to pass instead of strict 8/8. Preserves curved surface sheets while still rejecting non-surface voxels (75% evidence threshold).

### Geometric vs Structural Reconstruction
Documented the two competing approaches:
1. **Geometric reconstruction:** Match the original TO mesh shape as closely as possible (surface fidelity)
2. **Structural reconstruction:** Match the structural properties (stiffness, load paths) — may not look identical but performs the same

Concluded: The pipeline should primarily target structural reconstruction, with geometric fidelity as a secondary objective. The parametric output enables downstream design refinement that pure geometric matching cannot.

### Weekly Summary
**Key Achievements:** Root cause of curved surface loss identified and fix designed. Clear articulation of contributions.
**Reflection:** The curved surface bug was subtle — strict octant matching works for flat surfaces but fails for any curvature. The relaxed threshold (6/8) is a principled compromise.

---

## Easter Break — 17 March – 6 April 2026

### Development and Report Writing

**Curved Surface Fix Implementation:**
- Implemented `is_surface_point_relaxed(hood, min_octants=6)` in `thinning.py` for mode=3 only
- Validation on synthetic half-cylinder (R=15, thickness=5):

| Metric | Strict (8/8) | Relaxed (6/8) |
|--------|-------------|---------------|
| mode=3 voxels | 277 | 868 |
| Two-pass difference | 155 | 730 |

3× improvement in curved surface preservation. Flat slab (20×20×4) still correct: 256→416 voxels (more coverage, same 1-voxel thickness).

**Shell FEM Implementation:**
- Added shell finite element formulation using Mindlin-Reissner shell theory (accounts for transverse shear)
- Integration with existing beam FEM for hybrid structures
- Plates analysed with shell elements, beams with Euler-Bernoulli frame elements

**Additional Features:**
- Symmetry boundary condition support for reduced computational domains
- Fused body export for FreeCAD (single solid)
- `is_curved` flag per plate and vertex normal computation for mid-surface representation
- `_compute_vertex_normals()` helper in plate_extraction.py

**FreeCAD Curved Shell Rendering:**
- `try_shell_from_midsurface()`: Offset mid-surface by ±thickness/2 along vertex normals
- Creates inner/outer surfaces + stitches boundary edges
- Routed first for `is_curved=True` plates; flat plates use existing extruded voxels

**Report Writing:**
- Drafted results chapter with test case comparisons
- Created publication-quality figures showing pipeline stages
- Wrote discussion sections analysing pipeline strengths and limitations

### Summary
The Easter break was used intensively for both technical development and report writing. The curved surface fix and shell FEM represent the final technical contributions.

---

## Week 20 — 7–13 April 2026

### Supervisor Meeting (13 Apr)

**Questions Prepared:**
- Does the hybrid and curved extension flow well in the report?
- Which results are most impactful?
- Is the depth of critical analysis sufficient?

**Discussion Notes:**
- Supervisor approved report structure
- Discussed which figures best demonstrate pipeline capabilities
- Agreed on final list of results
- Discussion on dissertation feedback notes:
  - Consider making the `--mesh_input` section smaller — just mention it can run without size/layout optimisation with a quick example
  - Rewrite references to "surfaces" not just "plates" throughout
  - Make all diagrams unified in colour scheme
  - Improve introduction paradigm figure
  - Change cantilever to 150×40×4
  - Add references to flowchart figures
  - Justify SIMP in introduction (but note same methods could apply to level-set — TO-method agnostic)
  - Add section about curved/straight beam interaction
  - Report needs critical evaluation — advantages, disadvantages, what worked and didn't

**Questions to Attach with Dissertation Draft:**
1. Not sure about best way to display CAD figures (currently Solidworks matte titanium finish)
2. Research Gap and Justification section — guidance on framing
3. Code Availability section — want to reference GitHub and documentation
4. Figure quality, especially results section
5. May replace all compliance values with proper SimScale FEA values

**Actions Agreed:**
- Complete results chapter
- Write conclusion with future work
- Final proofread and formatting pass
- Compile and submit logbook

### Major Commit (15 Apr)
Full dissertation pipeline integration: shell FEM, symmetry BCs, fused body export, two-pass classification, comprehensive documentation.

### Weekly Summary
**Focus:** Final integration and report completion.
**Work Completed:**
- Integrated all pipeline components into single clean codebase
- Completed results chapter with quantitative comparisons:
  - Cantilever beam (150×50×4): baseline beam-only
  - Bridge: mixed beam-plate classification
  - Roof: plate-dominated with supporting beams
  - Quadcopter: branching topology with multiple load cases
- Critical discussion of limitations (minimum feature size, curved FEM reliability, block-size dependency)
- Generated final figures with consistent formatting

**Key Achievements:** Complete, working pipeline from TO density field to parametric CAD model with automatic beam-plate classification, structural optimisation, and FreeCAD export.

**Reflection:** Looking back at the project as a whole, the key contributions are:
1. Multi-signal zone classification that robustly handles ambiguous geometries
2. Two-pass topological classification requiring no parameter tuning
3. Complete automated pipeline with FreeCAD integration
4. Curved surface preservation fix for voxel-grid discretisation effects

The project evolved significantly from the original plan — the hybrid beam-plate handling emerged as a natural and necessary extension.

---

## Week 21 — 14–15 April 2026

### Weekly Summary
**Focus:** Report finalisation and logbook compilation.
**Work Completed:**
- Final report editing and proofreading
- Logbook compiled and organised for submission
- Last pipeline refinements and code cleanup
- Verified all test cases produce clean results for appendix

---

## Appendix A — Test Case Summary

| Test Case | Domain Size | Vol. Frac. | Classification | Key Feature |
|-----------|------------|------------|----------------|-------------|
| Cantilever Beam | 150×50×4 | 0.10 | All beam | Baseline validation |
| Bridge | 60×30×30 | 0.10 | Mixed (beam + plate) | Joint detection |
| Roof Structure | 60×60×30 | 0.10 | Plate-dominated | Surface preservation |
| Quadcopter | 40×40×4 | 0.10 | All beam (branching) | Multi-load case |

## Appendix B — Key Design Decisions Log

| Decision | Date | Rationale |
|----------|------|-----------|
| Skeleton-based over surface-based approach | Oct 2025 | Produces interpretable parametric output (beams/joints) |
| FreeCAD over Solidworks API | Dec 2025 | Open-source, Python-scriptable, no licence restrictions |
| Competition Classification (Westin metrics) | Feb 2026 | Automatic plate/rod distinction without manual threshold |
| Deep Capture strategy (Competition + Dilation + Propagation) | Feb 2026 | Plate edge misclassification fixed via morphological processing |
| 24 rotational symmetries for surface patterns | Feb 2026 | Tilted plate preservation — incomplete rotations caused failures |
| Multi-signal zone classification (V2 scoring) | Feb 2026 | Single metrics unreliable for ambiguous geometries |
| Curved beams geometry-only (no FEM) | Feb 2026 | Short beam segments → singular stiffness matrices |
| Two-pass topological classification | Mar 2026 | Parameter-free, exploits mathematical thinning mode difference |
| Relaxed surface point detection (6/8) | Mar 2026 | Strict 8/8 fails on voxel-grid staircase patterns |
| E₀ = 1000 in SIMP | Feb 2026 | Match frame FEM for meaningful compliance comparison |
| Stress-informed pruning (explored) | Feb 2026 | Potential contribution — geometric heuristics miss critical load paths |

## Appendix C — Supervisor Meeting Attendance

| Week | Date | Attended | Key Topic |
|------|------|----------|-----------|
| 1 | 8 Oct | Yes | Project scoping, research directions |
| 2 | 14 Oct | Yes | Geometry engines, manual workflow |
| 4 | 30 Oct | Yes | Approach selection, project plan, user-friendliness |
| 5 | 6 Nov | Yes | Literature review feedback ("make it flow like a story") |
| 8 | ~25 Nov | Yes | Input format, reconstruction goal, plate handling |
| 9 | ~3 Dec | Yes | Report submission, Christmas plan |
| 10 | ~10 Dec | Yes | GitHub, test cases, "find best then optimise" |
| 12 | ~15 Jan | Yes | Brief check-in |
| 13 | ~29 Jan | Yes | TO solver discussion, term plan |
| 14 | 3 Feb | Yes | Term plan review, pipeline priorities |
| 15 | ~12 Feb | Yes | Pipeline demo, zone classification, hybrid inputs |
| 17 | 3 Mar | Yes | Results review, notation/E₀ consistency |
| 18 | ~5 Mar | Yes | Report structure, TexText, convergence details |
| 19 | 13 Mar | Yes | Contributions, two-pass classification, results selection |
| 20 | 13 Apr | Yes | Final results, report structure, figure quality |

## Appendix D — OVERVIEW: Frame/Beam Based Reconstruction Pipeline

**Context:** Builds on Yin, Xiao & Cirak (2020) — "Topologically robust CAD model generation for structural optimisation."

**Pipeline Architecture — 4 Stages:**

**Stage 0 — SIMP Topology Optimisation:**
Input: domain dimensions, boundary conditions, volume fraction. Output: 3D binary voxel field. Method: Solid Isotropic Material with Penalisation (SIMP) with density filtering and sensitivity filtering. BC tagging: tag=1 (fixed), tag=2 (loaded), tag=3 (passive void).

**Stage 1 — Skeleton Reconstruction:**
1.1 Homotopic thinning (mode 0: curve-preserving, mode 1: surface-preserving, mode 3: combined)
1.2 Graph extraction: skeleton voxels → NetworkX graph → chain decomposition
1.3 Graph cleanup: prune short branches, merge close nodes, RDP simplification
1.4 Radius estimation: EDT (local thickness) or Volume Matching (uniform)

**Stage 2 — Layout Optimisation:**
3D Euler-Bernoulli frame analysis. Minimise compliance by adjusting node positions. L-BFGS-B optimiser with bounds.

**Stage 3 — Size Optimisation:**
Optimality Criteria (OC) update for cross-sectional areas. Volume constraint maintained. Beam element stiffness scales with area/moment of inertia.

**Final Output:** JSON with node positions, edge connectivity, radii, control points (if curved), plate meshes with thickness. FreeCAD macro generates parametric CAD model.

## Appendix E — Revised Project Plan (Full)

**Research Questions:**
- RQ1: Can homotopic thinning be extended to preserve plate-like structures?
- RQ2: How should beams and plates be classified robustly from voxel density fields?
- RQ3: Can size/layout optimisation be generalised to mixed beam-shell structures?
- RQ4: Does the reconstructed parametric model preserve structural performance?
- RQ5: Can the pipeline run fully automatically with sensible defaults?
- RQ6: How does reconstruction quality depend on TO resolution and volume fraction?
