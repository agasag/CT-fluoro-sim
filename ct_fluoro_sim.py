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
import nibabel

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
# --- (1a) LOAD DATA ---
# Path to the folder containing DICOM slices (eor path to nifti file (incl. -> .nii.gz)
data_dir = "E:/s0088/ct.nii.gz" #"E:/s0088/ct.nii.gz" #"E:/LIDC/NBIA/LIDC-IDRI-0004/01-01-2000-NA-NA-91780/3000534.000000-NA-58228/"
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
    ct_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    spacing = sitk_image.GetSpacing()[::-1]
elif data_dir_type == 'nifti':
    print('=== nifti ===')
    sitk_image = nibabel.load(data_dir)
    ct_array = sitk_image.get_fdata()
    affine = sitk_image.affine
    img_header = sitk_image.header
    spacing = img_header['pixdim']
    spacing = spacing[1:4]


ct_array = np.flip(ct_array, axis=(0,2))  # flip Y and X (todo: check if it's neccessary) # 241, 512, 512

print(f"====== CT volume shape (Z, Y, X): {ct_array.shape}")
# --- (1b) LOAD SEGMENTATIONS (to limt ROI) ---
seg_paths = {
    'aorta': "aorta.nii.gz",
    'right_atrium': "heart_atrium_right.nii.gz",
    'left_atrium': "heart_atrium_left.nii.gz",
    'right_ventricle': "heart_ventricle_left.nii.gz",
    'left_ventricle': "heart_ventricle_right.nii.gz",
    'heart_myocardium': "heart_myocardium.nii.gz"
}

# Initialize combined mask as zeros
combined_mask = np.zeros(ct_array.shape, dtype=bool)

# Load each segmentation and combine
for name, path in seg_paths.items():
    if data_dir_type == 'dicom':
        seg_img = sitk.ReadImage(data_dir.replace('NBIA', 'NBIA_totalsegmentator') + '/' + path)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(bool)
        combined_mask = combined_mask | seg_arr
    elif data_dir_type == 'nifti':
        seg_img_path = os.path.join(os.path.dirname(data_dir), "segmentations") + '/' + path
        seg_img = nibabel.load(seg_img_path)
        seg_arr = seg_img.get_fdata().astype(bool)
        combined_mask = combined_mask | seg_arr

# Now combined_mask is True where any structure exists

# Find bounding box of combined structures

# Initialize combined mask as zeros
padding_mm = 0  # extra physical padding around bounding box
  # spacing in (z, y, x)
z_spacing, y_spacing, x_spacing = spacing
shape = ct_array.shape  # (Z, Y, X)

# --- Get bounding box from mask ---
coords = np.argwhere(combined_mask)
z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0)

# --- Convert padding in mm to voxels ---
pad_x = int(padding_mm / x_spacing)
pad_y = int(padding_mm / y_spacing)
pad_z = int(padding_mm / z_spacing)

# --- Pad the bounding box ---
"""x_min = max(x_min - pad_x, 0)
x_max = min(x_max + pad_x, shape[2] - 1)

y_min = max(y_min - pad_y, 0)
y_max = min(y_max + pad_y, shape[1] - 1)

z_min = max(z_min - pad_z, 0)
z_max = min(z_max + pad_z, shape[0] - 1)
"""
# --- Determine square size ---
width = x_max - x_min + 1
height = y_max - y_min + 1
side = max(width, height)
print(side)

# --- Compute extra padding to make it square ---
extra_x = side - width
extra_y = side - height

x_min = max(x_min - extra_x // 2, 0)
x_max = min(x_max + (extra_x - extra_x // 2), shape[2] - 1)

y_min = max(y_min - extra_y // 2, 0)
y_max = min(y_max + (extra_y - extra_y // 2), shape[1] - 1)

# --- Crop CT and mask ---
ct_roi = ct_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
mask_roi = combined_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

# --- Force square slices after crop ---
z, h, w = ct_roi.shape
if h != w:
    pad_diff = abs(h - w)
    if h < w:
        # Pad height (Y) axis
        pad_top = pad_diff // 2
        pad_bottom = pad_diff - pad_top
        ct_roi = np.pad(ct_roi, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        mask_roi = np.pad(mask_roi, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    else:
        # Pad width (X) axis
        pad_left = pad_diff // 2
        pad_right = pad_diff - pad_left
        ct_roi = np.pad(ct_roi, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        mask_roi = np.pad(mask_roi, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

# --- Sanity check ---
print("Final shape (Z, Y, X):", ct_roi.shape)
assert ct_roi.shape[1] == ct_roi.shape[2], "Slices are not square!"

# Replace original array with cropped one
ct_array = ct_roi


##########################################################################################
# --- (2) CONVERT HU to mu ---
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


#print("Voxel spacing (x,y,z):", spacing)
# --- Step 2: ASTRA Volume geometry (note reversed axes order: X, Y, Z) ---
"""vol_geom = astra.create_vol_geom(
    int(ct_array.shape[2]),
    int(ct_array.shape[1]),
    int(ct_array.shape[0])
)"""

##########################################################################################
# --- (3) VOL GEOMETRY (Astra toolbox) --- (todo: check if reversed axis order isn't a bug)
#mu_array = np.transpose(mu_array, (2, 1, 0))  # (X, Y, Z)
# get spacing
#spacing = sitk_image.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
x_spacing, y_spacing, z_spacing = spacing[0], spacing[1], spacing[2]
X, Y, Z = mu_array.shape
vol_geom = astra.create_vol_geom(
     Z, Y, X,
        -Z * z_spacing / 2, Z * z_spacing / 2,
        -Y * y_spacing / 2, Y * y_spacing / 2,
        -X * x_spacing / 2, X * x_spacing / 2)

vol_id = astra.data3d.create('-vol', vol_geom, data=mu_array)
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
gamma = 0.5  # < 1 increases contrast in dark areas (bone)
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
