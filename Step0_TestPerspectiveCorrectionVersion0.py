import cv2
import numpy as np
import os

output_folder = "../all_square_aligned"
os.makedirs(output_folder, exist_ok=True)

input_folder = '../a'
files = os.listdir(input_folder)

pixels_per_mm = 25  # for example
anchor_mm = 14
expected_px = anchor_mm * pixels_per_mm  # 350 px

# LAB distance threshold
LAB_DISTANCE_THRESHOLD = 25 # smaller threshold for precise detection

# Morphology kernel
kernel = np.ones((5,5), np.uint8)

# Reference RGB colors for the square anchors (e.g., pink)
reference_rgbs = np.array([
    [113, 81, 68], [200, 148, 131], [88, 122, 159], [88, 108, 67],
    [128, 129, 178], [87, 192, 175], [227, 125, 51], [66, 90, 172],
    [198, 82, 99], [91, 60, 108], [158, 191, 68], [231, 163, 48],
    [44, 62, 147], [62, 149, 77], [180, 48, 57], [240, 201, 46],
    [194, 85, 155], [0, 137, 173], [236, 235, 236], [203, 206, 208],
    [161, 164, 168], [119, 121, 124], [82, 83, 89], [50, 50, 51]
], dtype=np.float32)

def order_points(pts):
    """
    Orders coordinates: top-left, top-right, bottom-right, bottom-left
    """    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL    
    rect[2] = pts[np.argmax(s)] # BR    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR    
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def save_debug_image(square_cnt, img, image_name=None, rgb=None):
    # square_cnt: contour, may not be exactly 4 points
    debug_img = img.copy()
    # Draw all contours
    if square_cnt is not None:
        cv2.drawContours(debug_img, [square_cnt], -1, (0,0,255), 2)
        # Draw points if it has 4
        if len(square_cnt) == 4:
            for i, (x, y) in enumerate(square_cnt.reshape(4,2)):
                cv2.putText(debug_img, f"{i}", (x-10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    # Save
    # if idx is None:
    debug_path = os.path.join(output_folder, f"square_{rgb}_{image_name}")
    # else:
    #     debug_path = os.path.join(output_folder, f"square_{image_name}_{idx}.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"Saved debug image: {debug_path}")

def build_straight_paper_dst(corners):
    """
    corners: (4,2) ordered TL,TR,BR,BL (SOURCE patch corners)
    returns: (4,2) destination corners TL,TR,BR,BL
    """
    tl, tr, br, bl = corners

    max_x = max(tl[0], bl[0], tr[0], br[0])
    min_x = min(tl[0], bl[0], tr[0], br[0])
    max_y = max(tl[1], bl[1], tr[1], br[1])
    min_y = min(tl[1], bl[1], tr[1], br[1])

    tl_dst = np.array([min_x, min_y], dtype=np.float32)
    tr_dst = np.array([max_x, min_y], dtype=np.float32)
    bl_dst = np.array([min_x, max_y], dtype=np.float32)
    br_dst = np.array([max_x, max_y], dtype=np.float32)

    return np.array([tl_dst, tr_dst, br_dst, bl_dst], dtype=np.float32)

def process_and_verify(input_image, image_name):

    squares_src_pts = []
    squares_cnt = []
    squares_dst_pts = []

    # --- STEP 1: PREPROCESS ---    
    img = input_image  
    H_img, W_img = img.shape[:2]

    # Convert to LAB
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    for rgb in reference_rgbs:
        # Convert color(e.g., pink) reference to LAB
        lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]
        # Euclidean distance in LAB
        diff = np.linalg.norm(image_lab.astype(np.float32) - lab.astype(np.float32), axis=2)
        color_mask = (diff < LAB_DISTANCE_THRESHOLD).astype(np.uint8) * 255
           
        # Morphology
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ No {rgb} detected in {image_name}")
            continue

        if len(contours) == 1:
            print(f"Found single color region for {rgb} in {image_name}")
            # append to squares_cnt
            squares_cnt.append((contours[0], rgb))
            # break  # Found single color region, proceed
        else:
            print(f"Multiple regions found for {rgb} in {image_name}, trying next color.")
            continue  # Try next color
    
    # # Save all contour for debugging
    # square_cnt = None
    # for idx, c in enumerate(contours):
    #     perimeter = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    #     print(f"Contour {idx}: vertices={len(approx)}, area={cv2.contourArea(c)}")
    #     save_debug_image(approx, img, idx=idx, image_name=image_name)

    # --- STEP 2: DETECT THE SQUARE (ANCHOR) --- 
    for square_cnt, rgb in squares_cnt:
        perimeter = cv2.arcLength(square_cnt, True)
        approx = cv2.approxPolyDP(square_cnt, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)        
        aspect_ratio = float(w)/h
        if len(approx) == 4 and 0.8 < aspect_ratio < 1.5 and cv2.contourArea(square_cnt) > 1000:
            
            save_debug_image(square_cnt, img, image_name=image_name, rgb=rgb.tolist())

            src_pts = order_points(approx.reshape(4, 2))
            dst_pts = build_straight_paper_dst(src_pts)

            squares_src_pts.append(src_pts)
            squares_dst_pts.append(dst_pts)
        
        # # Logic: 4 corners, square-ish shape, and located generally 'top' or valid size
        # if len(approx) == 4 and 0.8 < aspect_ratio < 1.5 and cv2.contourArea(square_cnt) > 1000:
        #     print(f"Aspect Ratio: {aspect_ratio:.2f}, vertices={len(approx)}, area={cv2.contourArea(square_cnt)}")
        #     save_debug_image(approx, img, image_name=image_name, rgb=rgb.tolist())

    if len(squares_src_pts) == 0:
        print("Failed: Could not detect the 14mm anchor square.")
        return img
    
    # --- STEP 3: WARP PERSPECTIVE ---    
    squares_src_pts= np.vstack(squares_src_pts).astype(np.float32)
    squares_dst_pts= np.vstack(squares_dst_pts).astype(np.float32)

    matrix, _ = cv2.findHomography(squares_src_pts, squares_dst_pts, cv2.RANSAC, 3.0)
    
    # matrix = cv2.getPerspectiveTransform(squares_src_pts, squares_dst_pts)

    corners = np.array([[0,0],[W_img,0],[W_img,H_img],[0,H_img]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, matrix)
    xs, ys = warped_corners[:,0,0], warped_corners[:,0,1]
    tx, ty = -xs.min() if xs.min()<0 else 0, -ys.min() if ys.min()<0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    final_matrix = T @ matrix
    out_w, out_h = int(np.ceil(xs.max() + tx)), int(np.ceil(ys.max() + ty))

    warped = cv2.warpPerspective(img, final_matrix, (out_w, out_h))
    
    # # --- STEP 4: REFINED VERIFICATION (Rotated Rect) ---    
    # warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)   
    # _, warped_thresh = cv2.threshold(warped_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # warped_contours, _ = cv2.findContours(warped_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    # line_found = False
    
    # for c in warped_contours:
    #     # Get the Rotated Rectangle
    #     # rect returns ((center_x, center_y), (width, height), angle)        
    #     rect = cv2.minAreaRect(c)
    #     (center_x, center_y), (dim1, dim2), angle = rect
    
    #     # Determine which dimension is length (max) and thickness (min)
    #     length_px = max(dim1, dim2)
    #     thickness_px = min(dim1, dim2)
    
    #     # Filter Logic:
    #     # 1. Location: Bottom half (center_y > 500)
    #     # 2. Shape: Must be long and thin (Ratio > 3:1)
    #     if center_y > 500 and length_px > thickness_px * 3 and length_px > 50:
    #         measured_mm = length_px / pixels_per_mm
    #         expected_mm = 20.0
    #         error = abs(measured_mm - expected_mm)
        
    #         # --- Visualization ---# Convert rotated rect to 4 points to draw it
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)
            
    #         # Green if error < 0.5mm, else Red
    #         color = (0, 255, 0) if error < 0.5 else (0, 0, 255)
    #         cv2.drawContours(warped, [box], 0, color, 2)
            
    #         label = f"{measured_mm:.2f}mm (Err: {error:.2f})"
    #         cv2.putText(warped, label, (int(center_x) - 50, int(center_y) - 20), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    #         print(f"--- Verified {image_path} ---")
    #         print(f"Length: {measured_mm:.2f} mm | Expected: 20 mm | Error: {error:.2f} mm")
        
    #         line_found = True
    #         break 
    #     # Stop after finding the first valid line
    #     if not line_found:
    #         print("Warning: Square corrected, but no valid line found in bottom area.")

    return warped

# --- MAIN PROCESSING LOOP ---
image_extensions = (".jpg", ".JPG", ".jpeg", ".png", ".tif", ".tiff")

files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(image_extensions)
])

for file in files:
    print(f"Processing: {file}")
    path = os.path.join(input_folder, file)

    img = cv2.imread(path)
    if img is None:
        print(f"❌ Could not read {file}")
        continue

    result = process_and_verify(img, file)

    out_path = os.path.join(output_folder, file)
    cv2.imwrite(out_path, result)
    print(f"✅ Saved corrected image to {out_path}")

    # process_and_verify('../a/Faba-Seed-CC_Vf5-3-1.JPG', '../square_aligned/result.jpg')