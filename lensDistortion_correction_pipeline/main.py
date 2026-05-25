import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

# INPUT_DIR = "Data/"
INPUT_DIR = "../../../../../../data/phenomics_images/faba_images"
OUTPUT_DIR = "perspective_lens_output/images/"

def calibrate(showPics=True):

    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages//calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    nRows = 9
    nCols = 6
    termCriteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:

        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)

        cornersFound, cornersOrg = cv.findChessboardCorners(
            imgGray,
            (nRows, nCols),
            None
        )

        if cornersFound == True:

            worldPtsList.append(worldPtsCur)

            cornersRefined = cv.cornerSubPix(
                imgGray,
                cornersOrg,
                (11, 11),
                (-1, -1),
                termCriteria
            )

            imgPtsList.append(cornersRefined)

            if showPics:
                cv.drawChessboardCorners(
                    imgBGR,
                    (nRows, nCols),
                    cornersRefined,
                    cornersFound
                )

                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)

    cv.destroyAllWindows()

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList,
        imgPtsList,
        imgGray.shape[::-1],
        None,
        None
    )

    print('Camera Matrix:\n', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration Parameters (later video)
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')

    np.savez(
        paramPath,
        repError=repError,
        camMatrix=camMatrix,
        distCoeff=distCoeff,
        rvecs=rvecs,
        tvecs=tvecs
    )

    return camMatrix, distCoeff


def removeDistortion(imgPath, filename, camMatrix, distCoeff):
    img = cv.imread(imgPath)
    height, width = img.shape[:2]
    newCamMatrix, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (width, height), 1, (width, height))
    imgUndistorted = cv.undistort(img, camMatrix, distCoeff, None, newCamMatrix)

    # # Draw line to see Distortion change
    # cv.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 2)
    # cv.line(imgUndistorted, (0, height // 2), (width, height // 2), (0, 255, 0), 2)

    # save the undistorted image
    cv.imwrite(os.path.join(OUTPUT_DIR, f"undistorted_{filename}"), imgUndistorted)

if __name__ == '__main__':
    
    # 1. create the camMatrix manualy
    camMatrix = np.array([[4035, 0, 3000], [0, 4035, 2000], [0, 0, 1]])

    # 2. create the distCoeff manualy
    distCoeff = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])

    # 3. read all images from Data folder and remove distortion on all and save images
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith((".jpg", ".JPG")):
            imgPath = os.path.join(INPUT_DIR, filename)
            removeDistortion(imgPath, filename, camMatrix, distCoeff)

#### GIMP Simulation ------------------------
# import cv2
# import numpy as np
# import os

# # --- GIMP SLIDER VALUES ---
# MAIN = -9.37  
# EDGE = 2.99    
# ZOOM = -7.01  
# # --------------------------

# INPUT_DIR = "Data/"
# OUTPUT_DIR = "perspective_lens_output/images/"

# def apply_gimp_distortion_fixed(image_path, output_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Image not found.")
#         return

#     h, w = img.shape[:2]
#     cx, cy = w / 2.0, h / 2.0
    
#     # GIMP uses the distance to the farthest corner as the normalization unit (r=1.0)
#     max_radius = np.sqrt(cx**2 + cy**2)

#     # 1. Create the grid
#     y, x = np.indices((h, w), dtype=np.float32)

#     # 2. Calculate displacement from center
#     dx = x - cx
#     dy = y - cy
    
#     # 3. Calculate normalized radius (r)
#     r = np.sqrt(dx**2 + dy**2) / max_radius
#     r2 = r**2
#     r4 = r2**2

#     # 4. GIMP's internal polynomial scaling
#     # The '1.0' at the start keeps the image intact; the variables distort it.
#     mag = (1.0 + (MAIN / 200.0) * r2 + (EDGE / 200.0) * r4)

#     # 5. Zoom (Rescale) Logic
#     # GIMP's zoom is inverse: negative moves 'in', positive moves 'out'
#     zoom_factor = 1.0 - (ZOOM / 100.0)

#     # 6. Generate the maps and FORCE float32
#     # This is the line that was causing your error.
#     map_x = (cx + dx * mag * zoom_factor).astype(np.float32)
#     map_y = (cy + dy * mag * zoom_factor).astype(np.float32)

#     # 7. Remap with high-quality cubic interpolation
#     result = cv2.remap(
#         img, 
#         map_x, 
#         map_y, 
#         interpolation=cv2.INTER_CUBIC, 
#         borderMode=cv2.BORDER_CONSTANT
#     )

#     cv2.imwrite(output_path, result)
#     print(f"Success! Saved to {output_path}")

# # Run the function on all images in the Data path
# for filename in os.listdir(INPUT_DIR):
#     if filename.endswith((".jpg", ".JPG")):
#         input_path = os.path.join(INPUT_DIR, filename)
#         output_path = os.path.join(OUTPUT_DIR, f"corrected_{filename}")
#         apply_gimp_distortion_fixed(input_path, output_path)
