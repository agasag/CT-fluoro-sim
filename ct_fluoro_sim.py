'''
CT-fluoro-sim
Code to run a fluoroscopy simulation

inputs: 3D CT volume
outputs: 2D fluoroscopy-like images at given angles

'''

import astra
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import cv2

def detect_image_type(path):
    """ check whether a file is NIfTi img or a folder with DICOMs; else: unknown"""
    # Check for NIfTI file
    if os.path.isfile(path) and path.lower().endswith(('.nii', '.nii.gz')):
        return "nifti"

    # Check for DICOM series folder
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(path)
        if series_IDs:
            return "dicom"

    return "unknown"

##########################################################################################
# --- (1) LOAD DATA ---
# Path to the folder containing DICOM slices or path to nifti file
data_dir = "E:/LIDC/NBIA/LIDC-IDRI-0004/01-01-2000-NA-NA-91780/3000534.000000-NA-58228/"
data_dir_type = detect_image_type(data_dir)

if data_dir_type == 'dicom':
    print('=== dicom ===')

    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(data_dir)

    if not series_IDs:
        raise Exception("No DICOM series found in the directory.")

    # Choose the first series (most cases have only one)
    series_file_names = reader.GetGDCMSeriesFileNames(data_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)

    # Read the series into a 3D volume
    sitk_image = reader.Execute()
elif data_dir_type == 'nifti':
    print('=== nifti ===')
    sitk_image = sitk.ReadImage(data_dir)

##########################################################################################
# --- (2) LOAD CT AND CONVERT HU to mu ---

ct_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
ct_array = np.flip(ct_array, axis=(0,2))  # flip Y and X (todo: check if it's neccessary)

print(f"====== CT volume shape (Z, Y, X): {ct_array.shape}")

# Convert HU to linear attenuation coefficient (approximate)
mu_water = 0.02  # cm^-1 at ~70 keV
#mu_array = mu_water * (1 + ct_array / 1000)# (ct_array + 1000) / 1000 * 0.2  # [cm^-1], approx scaling

# commented
#mu_array = np.clip((ct_array + 1000) / 1000 * 0.02, 0, 0.05)

# segment into Air / Soft Tissue / Bone ---
labels = np.zeros_like(ct_array)
labels[ct_array < -300] = 0                        # Air
labels[(ct_array >= -300) & (ct_array < 200)] = 1  # Soft tissue
labels[ct_array >= 200] = 2                        # Bone

# assign Attenuation Coefficients (approx. for 70 keV) ---
mu_air = 0.0001
mu_soft = 0.17
mu_bone = 0.5
mu_array = (
    (labels == 0) * mu_air +
    (labels == 1) * mu_soft +
    (labels == 2) * mu_bone
).astype(np.float32)

# get spacing
spacing = sitk_image.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
#print("Voxel spacing (x,y,z):", spacing)
# --- Step 2: ASTRA Volume geometry (note reversed axes order: X, Y, Z) ---
"""vol_geom = astra.create_vol_geom(
    int(ct_array.shape[2]),
    int(ct_array.shape[1]),
    int(ct_array.shape[0])
)"""

##########################################################################################
# --- (3) VOL GEOMETRY (Astra toolbox) --- (todo: check if reversed axis order isn't a bug)
x_spacing, y_spacing, z_spacing = spacing[0], spacing[1], spacing[2]

vol_geom = astra.create_vol_geom(
    int(ct_array.shape[2]),
    int(ct_array.shape[1]),
    int(ct_array.shape[0]),
    -int(ct_array.shape[2])* x_spacing / 2,  int(ct_array.shape[2]) * x_spacing / 2,  # X
    -int(ct_array.shape[1]) * y_spacing / 2,  int(ct_array.shape[1]) * y_spacing / 2,  # Y
    -int(ct_array.shape[0])* z_spacing / 2,  int(ct_array.shape[0]) * z_spacing / 2   # Z
)


##########################################################################################
# --- (4) PROJECTION ANGLES ---
num_angles = 360
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)
print(f"====== Number of angles: {len(angles)}")

##########################################################################################
# --- (5) PROJECTION GEOMETRY (cone-beam) ---
# Detector parameters
det_row_count = 512
det_col_count = 512
det_spacing = 1

# Geometry distances (todo: check if it's correct + get directly from dicom tags)
source_origin_dist = 1085.5 #  # mm
origin_detector_dist =  595.0  # mm

# prepare geometry
proj_geom = astra.create_proj_geom(
    'cone',
    det_spacing, det_spacing, #det_spacing, det_spacing,
    det_col_count, det_row_count,
    angles,
    source_origin_dist,
    origin_detector_dist
)

##########################################################################################
# --- (6) CONFIGURE FORWARD PROJECTION (CUDA used) ---
vol_id = astra.data3d.create('-vol', vol_geom, data=mu_array)
proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
sino_id = astra.data3d.create('-sino', proj_geom)

cfg = astra.astra_dict('FP3D_CUDA')
cfg['VolumeDataId'] = vol_id
cfg['ProjectionDataId'] = sino_id
cfg['ProjectorId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

##########################################################################################
# --- (7) GET SINOGRAM ---
sinogram = astra.data3d.get(sino_id)
print(f"====== Sinogram shape: {sinogram.shape}")  # Expect (num_angles, det_row_count, det_col_count)

##########################################################################################
# --- (8) GET 2D FLUOROSCOPY ---
# select angle
chosen_angle_index = 160
line_integral = sinogram[:, chosen_angle_index, :]  # shape (det_row_count, angles, det_col_count) todo: why not (det, det, angles)
print(f"====== Extracted projection shape: {line_integral.shape}")
line_integral *= 0.01

##########################################################################################
# --- (9) SIMULATE FLUOROSCOPY, i.e. Beer-Lambert law + todo noise? + gamma correction? todo! ---

mA = 500  #567 # todo: get it from dicom tags / find regulations / find avg values
exposure_time_s = 0.005 # 5 ms # todo: get it from dicom tags / find regulations / find avg values
mAs = mA * exposure_time_s  # mAs
I0 = mAs * 1e7  # ~10^7 photons/mAs/mm^2

I = I0 * np.exp(-line_integral)
I_noisy = I  # No noise added

##########################################################################################
# --- (10) log + normalize
drr = -np.log(I_noisy / I0 + 1e-10)  #  attenuation map ---> according to beer-lambert law
drr_clipped = drr#np.clip(drr, 0, np.percentile(drr, 99))
drr_norm = (drr_clipped - np.min(drr_clipped)) / (np.max(drr_clipped) - np.min(drr_clipped))

print(f"====== 2D normalized image (result) shape: {drr_norm.shape}")

# normalized image, non-negative (i.e. bones are dark)
drr_positive = 1.0 - drr_norm  # invert grayscale

# gamma correction
gamma = 0.7  # < 1 increases contrast in dark areas (bone)
drr_gamma = drr_positive**gamma

# CLAHE - todo: is it efficient for our case?
drr_uint8 = (drr_positive * 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
drr_clahe = clahe.apply(drr_uint8)

##########################################################################################
# --- (11) Display ---

figure, axis = plt.subplots(1, 4)

axis[0].imshow(drr_clipped, cmap='gray')
axis[0].set_title("DRR")

axis[1].imshow(drr_clipped, cmap='gray')
axis[1].set_title("DRR - norm")

axis[2].imshow(drr_gamma, cmap='gray')
axis[2].set_title("DRR - positive")

axis[3].imshow(drr_clahe, cmap='gray')
axis[3].set_title("DRR - CLAHE")

##########################################################################################
# --- (12) Postprocessing ---
# todo: window/level adjustment? filtering?

# --- (13) Cleanup ---
astra.data3d.delete(vol_id)
astra.data3d.delete(sino_id)
astra.projector.delete(proj_id)
astra.algorithm.delete(alg_id)
