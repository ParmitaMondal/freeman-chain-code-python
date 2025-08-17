# freeman_chain_from_matlab_script.py
# Python translation of the MATLAB flow in your screenshot:
# 1) read image -> average filter (9x9, symmetric padding) -> Otsu threshold
# 2) extract OUTER boundary (bwboundaries) and visualize as a binary image
# 3) subsample boundary points and draw a polyline between them
# 4) compute Freeman chain code (8-connectivity), its first difference,
#    rotation-invariant minimal cyclic shift, and its first difference

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------
def imfilter_average(img, k=9):
    """Average filter with symmetric padding (like MATLAB 'symmetric')."""
    # OpenCV's BORDER_REFLECT_101 ~= MATLAB symmetric
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT_101)

def otsu_binarize(gray):
    """Otsu threshold -> binary (uint8 0/255)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def outer_boundary(binary_0_255):
    """Find the largest external contour (outer boundary) like bwboundaries."""
    cnts, _ = cv2.findContours(binary_0_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No boundary found.")
    areas = [cv2.contourArea(c) for c in cnts]
    k = int(np.argmax(areas))
    return cnts[k]  # Nx1x2

def bound2im(contour, M, N):
    """Rasterize a contour (polyline) into a binary image of size MxN."""
    canvas = np.zeros((M, N), np.uint8)
    cnt = contour.reshape(-1, 1, 2)
    cv2.polylines(canvas, [cnt], isClosed=True, color=255, thickness=1)
    return canvas

def subsample_polyline(contour, step=50):
    """Keep every 'step'-th vertex along the contour (ordered)."""
    pts = contour.reshape(-1, 2)
    if step <= 1 or pts.shape[0] <= step:
        sub = pts
    else:
        sub = pts[::step]
        # ensure closed loop
        if not np.array_equal(sub[0], sub[-1]):
            sub = np.vstack([sub, sub[0]])
    return sub

# Freeman 8-neighborhood (image coordinates: x→right, y→down)
DIRS = np.array([
    [ 1,  0],  # 0: E
    [ 1, -1],  # 1: NE
    [ 0, -1],  # 2: N
    [-1, -1],  # 3: NW
    [-1,  0],  # 4: W
    [-1,  1],  # 5: SW
    [ 0,  1],  # 6: S
    [ 1,  1],  # 7: SE
], dtype=np.int32)
DIR_LUT = {(dx, dy): i for i, (dx, dy) in enumerate(DIRS.tolist())}

def freeman_chain_code(points_xy):
    """Compute 8-connected Freeman chain code for an ordered closed polyline."""
    pts = np.round(points_xy).astype(np.int32)
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    steps = np.diff(pts, axis=0)
    steps = np.clip(steps, -1, 1)  # ensure unit steps
    code = []
    for dx, dy in steps:
        c = DIR_LUT.get((int(dx), int(dy)))
        if c is not None:
            code.append(c)
    return np.asarray(code, dtype=np.int32)

def first_difference(code):
    if code.size == 0:
        return code
    return (np.diff(code, append=code[0]) + 8) % 8

def minimal_cyclic_shift(code):
    """Return lexicographically minimal cyclic shift (simple O(n^2) is fine)."""
    if code.size == 0:
        return code
    best = code.copy()
    for s in range(1, len(code)):
        cand = np.roll(code, -s)
        # lexicographic compare
        for a, b in zip(cand, best):
            if a < b:
                best = cand
                break
            if a > b:
                break
    return best

# ---------------------------
# Main pipeline (mirrors MATLAB script)
# ---------------------------
if __name__ == "__main__":
    # f = imread('circular_stroke.tif');
    path = "circular_stroke.tif"  # change to your image path
    f = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if f is None:
        raise FileNotFoundError(path)

    # imshow(f)
    plt.figure(); plt.imshow(f, cmap="gray"); plt.title("I (input)"); plt.axis("off")

    # w = fspecial('average',9); g = imfilter(f,w,'symmetric');
    g = imfilter_average(f, k=9)
    plt.figure(); plt.imshow(g, cmap="gray"); plt.title("g (smoothed)"); plt.axis("off")

    # level = graythresh(f); BW = im2bw(g, level);  (Otsu on smoothed image)
    BW = otsu_binarize(g)
    plt.figure(); plt.imshow(BW, cmap="gray"); plt.title("BW (Otsu)"); plt.axis("off")

    # B = bwboundaries(BW); choose longest boundary B{k}
    b = outer_boundary(BW)

    # [M,N] = size(BW); g1 = bound2im(b,M,N);
    M, N = BW.shape
    g1 = bound2im(b, M, N)
    plt.figure(); plt.imshow(g1, cmap="gray"); plt.title("g1 (outer boundary raster)"); plt.axis("off")

    # [bs, sub] = bsubsamp(b, 50);  (subsample boundary)
    sub = subsample_polyline(b, step=50)
    # g2 = bound2im(sub, M, N);
    g2 = bound2im(sub.reshape(-1, 1, 2), M, N)
    plt.figure(); plt.imshow(g2, cmap="gray"); plt.title("g2 (subsampled boundary raster)"); plt.axis("off")

    # cn = connectpoly(S(:,1), S(:,2));  (draw polyline through subsampled points)
    vis = cv2.cvtColor(np.zeros_like(BW), cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [sub.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=1)
    plt.figure(); plt.imshow(vis[..., ::-1]); plt.title("Connected subsampled boundary"); plt.axis("off")

    # --- Freeman chain code for boundary (similar to fchcode) ---
    fcc = freeman_chain_code(sub)                 # c.fcc
    diff = first_difference(fcc)                  # c.diff
    c_mm = minimal_cyclic_shift(fcc)              # c.mm
    c_diffmm = first_difference(c_mm)             # c.diffmm
    x0y0 = tuple(sub[0])                          # c.x0y0

    print(f"Start (x0,y0): {x0y0}")
    print(f"Chain length: {len(fcc)}")
    print("fcc (first 40):     ", fcc[:40].tolist())
    print("diff (first 40):    ", diff[:40].tolist())
    print("c_mm (first 40):    ", c_mm[:40].tolist())
    print("c_diffmm (first 40):", c_diffmm[:40].tolist())

    plt.show()
