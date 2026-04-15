# Hybrid Beam-Plate Test Cases

## Purpose

These test cases create Top3D structures with **clearly distinguishable** beam and plate geometry to verify the hybrid zone classification is working correctly.

## Test Cases

### Test 1: Clear Beam-Plate Separation
**File**: `test_hybrid_clear.py`

**Geometry**:
- Domain: 40×40×20
- Distributed load on entire top face (forces plate formation)
- Fixed supports at 4 bottom corners (forces vertical beams)

**Expected Result**:
- ✓ Horizontal plate at top (z ≈ 18-20)
- ✓ Four vertical column beams at corners
- ✓ Clear EDT difference: plate thin (EDT ≈ 2), beams medium (EDT ≈ 3-4)

**Classification Expected**:
- Plate: High planarity (P > 0.85), low linearity (L < 0.3), low EDT
- Beams: High linearity (L > 0.6), low EDT

---

### Test 2: Cantilever with Plate End
**File**: `test_cantilever_plate.py`

**Geometry**:
- Domain: 60×10×10 (long and narrow)
- Fixed left face (x=0)
- Distributed load on right face (x=60)

**Expected Result**:
- ✓ Horizontal beam along length (x=0 to x≈50)
- ✓ Small vertical plate at right end to distribute load
- ✓ Spatial separation makes classification easy

**Classification Expected**:
- Beam: Linearity >> Planarity, medium EDT
- Plate: Planarity >> Linearity, low EDT

---

### Test 3: Single Thick Beam
**File**: `test_thick_beam.py`

**Geometry**:
- Domain: 50×15×15 (moderate length, thick cross-section)
- Volume fraction: 0.20 (allows thick beam)
- Point load at right center

**Expected Result**:
- ✓ Single thick beam along x-axis
- ✓ **High EDT** (3-5 voxels due to thick cross-section)
- ✓ Moderate linearity (thicker = less elongated)

**Critical Test**:
This tests the **thickness-aware classification fix**. Without it:
- High EDT + Moderate linearity → might look planar → misclassified as PLATE ❌

With thickness-aware fix:
- High EDT detected → threshold adjustment → classified as BEAM ✓

**Classification Expected**:
- Region should show: `beam (P=0.70, L=0.45, EDT=4.2)`
- High EDT should trigger thickness bias
- Should classify as BEAM despite moderate linearity

---

## Running the Tests

### Option 1: Run All Tests
```bash
python run_hybrid_tests.py
```

This will:
1. Run all 3 Top3D optimizations
2. Save .npz files to `output/hybrid_v2/`
3. Show expected results
4. Provide next steps

### Option 2: Run Individual Tests
```bash
# Test 1: Clear separation
python test_hybrid_clear.py

# Test 2: Cantilever
python test_cantilever_plate.py

# Test 3: Thick beam
python test_thick_beam.py
```

## Reconstruction

After generating the test cases, run reconstruction:

```bash
# Test 1
python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz \
  --hybrid --output Clear_Test.json --visualize \
  --planarity_thresh 0.20 \
  --linearity_thresh 0.80

python run_pipeline.py --skip_top3d --top3d_npz output/hybrid_v2/Clear_Beam_Plate_Test_top3d.npz --hybrid --output Clear_Test.json --visualize \
  --planarity_thresh 0.05 --linearity_thresh 0.95




# Test 2
python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/Cantilever_Plate_Test_top3d.npz \
  --hybrid --output Cantilever_Test.json --visualize

# Test 3 (Critical for thickness-aware fix)
python run_pipeline.py --skip_top3d \
  --top3d_npz output/hybrid_v2/Thick_Beam_Test_top3d.npz \
  --hybrid --output Thick_Beam_Test.json --visualize
```

## Verification

### 1. Console Output
Watch the zone classification output:

```
[1.5] Hybrid Mode: Pre-classifying zones (plate_threshold=8.2 voxels)...
    [ZoneClassifier] 3 thin regions to classify (min_size=160)
      Region 1: 8932 voxels -> plate (P=0.88, L=0.12, EDT=2.1)  ← Thin plate ✓
      Region 2: 5420 voxels -> beam (P=0.75, L=0.42, EDT=5.3)   ← Thick beam ✓
      Region 3: 4200 voxels -> beam (P=0.73, L=0.38, EDT=6.2)   ← Very thick beam ✓
```

**Key checks**:
- ✓ Low EDT (< 3) + High P → plate
- ✓ High EDT (> 4) + Moderate L → beam (thickness-aware fix working)
- ✓ High L (> 0.6) → beam

### 2. FreeCAD Visualization
Open the generated JSON in FreeCAD:

```python
# In FreeCAD:
# Macro → Execute → src/export/freecad_reconstruct_stable.py
# Select: output/hybrid_v2/Clear_Test.json
```

**Expected groups**:
- **Beams_CSG_Final** (red): Column beams, thick beams
- **Plates_CSG_Final** (cyan): Roof plate, end plates
- **Ref_Skeleton_Final** (yellow): Reference skeleton overlay

### 3. Visual Inspection
- ✓ Beams should be red cylinders with spheres at joints
- ✓ Plates should be cyan shells/surfaces
- ✓ No thick beams in the Plates group ← This was the bug!

## Debugging Classification Issues

If you see incorrect classifications:

### Thick beams classified as plates?
**Problem**: Thickness-aware fix not working

**Check**:
1. Verify fix is applied (check zone_classifier.py lines 199-204)
2. Check console shows EDT values
3. Try decreasing `--plate_thickness_ratio`:
   ```bash
   python run_pipeline.py ... --plate_thickness_ratio 0.10
   ```

### Plates classified as beams?
**Problem**: Thresholds too strict

**Check**:
1. Console shows plate has low planarity (P < 0.7)
2. Try adjusting defaults in zone_classifier.py:
   - Decrease `planarity_thresh` from 0.7 to 0.65
   - Increase `linearity_thresh` from 0.5 to 0.6

### Ambiguous classifications?
**Problem**: Shape metrics inconclusive

**Solutions**:
1. Increase domain size (more voxels for clearer PCA)
2. Decrease volume fraction (sparser = clearer shapes)
3. Adjust filter radius for smoother features

## Test Success Criteria

### Test 1 (Clear Separation)
- ✓ Exactly 1-2 plate regions (roof)
- ✓ Exactly 4 beam edges (columns)
- ✓ No thick columns in Plates group

### Test 2 (Cantilever)
- ✓ 1 beam edge (main cantilever)
- ✓ 0-1 plate region (end plate)
- ✓ Spatial separation clear in visualization

### Test 3 (Thick Beam)
- ✓ 1 beam edge (thick beam)
- ✓ 0 plate regions
- ✓ Console shows high EDT (> 4) with beam classification

## Date
Created: February 13, 2026
