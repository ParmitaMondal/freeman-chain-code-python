# Freeman Chain Code (8-connectivity) in Python

This repo provides a concise implementation of **Freeman's chain code** for a binary object boundary.
It follows a typical MATLAB-style workflow:

1. Read image (grayscale)
2. Smooth with a mean filter (9×9)
3. Otsu threshold → binary mask
4. Extract the **outer** boundary (largest contour)
5. Optionally subsample the boundary points
6. Compute:
   - `fcc`: Freeman chain code (0..7, 8-neighborhood)
   - `diff`: first difference (mod 8)
   - `c_mm`: rotation-invariant minimal-magnitude cyclic shift of `fcc`
   - `c_diffmm`: first difference of `c_mm` (mod 8)
   - `x0y0`: starting coordinate of the boundary

## Directions (Freeman 8-neighborhood)

We use the conventional mapping with image coordinates `(x, y)` where `x` increases to the right and `y` increases **down**:

