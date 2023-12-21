import concurrent.futures
import copy
import glob
import multiprocessing
import threading
import os
import re
import shutil
import pandas as pd
import psutil
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import partial
from time import time, sleep
import argparse
import json
import ndjson
import numpy as np
import scipy.stats
from dask.dataframe.multi import required
from scipy import ndimage, signal
from scipy.optimize import curve_fit
from tabulate import tabulate
from tqdm import tqdm
import pydicom
import nibabel as nib
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Use text styles in print
class ts:
	red = "\033[91m"
	yellow = "\033[93m"
	green = "\033[92m"
	cyan = "\033[96m"
	blue = "\033[94m"
	magenta = "\033[95m"
	
	ul = "\033[21m"  # underline
	underline = ul
	
	reset = "\033[0m"
	end = reset
	
	mb = lambda n: f"\033[{n}D"  # Move cursor back n characters


moose_labels = {1: "Adrenal-glands", 2: "Aorta", 3: "Bladder", 4: "Brain", 5: "Heart", 6: "Kidneys", 7: "Liver", 8: "Pancreas", 9: "Spleen", 10: "Thyroid", 11: "Inferior-vena-cava", 12: "Lung", 13: "Carpal", 14: "Clavicle", 15: "Femur", 16: "Fibula", 17: "Humerus", 18: "Metacarpal", 19: "Metatarsal", 20: "Patella", 21: "Pelvis", 22: "Phalanges-of-the-hand", 23: "Radius", 24: "Ribcage", 25: "Scapula", 26: "Skull", 27: "Spine", 28: "Sternum", 29: "Tarsal", 30: "Tibia", 31: "Phalanges-of-the-feet", 32: "Ulna", 33: "Skeletal-muscle", 34: "Subcutaneous-fat", 35: "Torso-fat", 36: "Psoas", 37: "R-Hippocampus", 38: "L-Hippocampus", 39: "R-Amygdala", 40: "L-Amygdala", 41: "R-Anterior-temporal-lobe-medial-part", 42: "L-Anterior-temporal-lobe-medial-part", 43: "R-Anterior-temporal-lobe-lateral-part", 44: "L-Anterior-temporal-lobe-lateral-part", 45: "R-Parahippocampal-and-ambient-gyri", 46: "L-Parahippocampal-and-ambient-gyri", 47: "R-Superior-temporal-gyrus-posterior-part", 48: "L-Superior-temporal-gyrus-posterior-part", 49: "R-Middle-and-inferior-temporal-gyrus", 50: "L-Middle-and-inferior-temporal-gyrus", 51: "R-Fusiform-gyrus", 52: "L-Fusiform-gyrus", 53: "R-Cerebellum", 54: "L-Cerebellum", 55: "Brainstem", 56: "L-Insula", 57: "R-Insula", 58: "L-Lateral-remainder-of-occipital-lobe", 59: "R-Lateral-remainder-of-occipital-lobe", 60: "L-Cingulate-gyrus-gyrus-cinguli-anterior-part", 61: "R-Cingulate-gyrus-gyrus-cinguli-anterior-part", 62: "L-Cingulate-gyrus-gyrus-cinguli-posterior-part", 63: "R-Cingulate-gyrus-gyrus-cinguli-posterior-part", 64: "L-Middle-frontal-gyrus", 65: "R-Middle-frontal-gyrus", 66: "L-Posterior-temporal-lobe", 67: "R-Posterior-temporal-lobe", 68: "L-Inferiolateral-remainder-of-parietal-lobe", 69: "R-Inferiolateral-remainder-of-parietal-lobe", 70: "L-Caudate-nucleus", 71: "R-Caudate-nucleus", 72: "L-Nucleus-accumbens", 73: "R-Nucleus-accumbens", 74: "L-Putamen", 75: "R-Putamen", 76: "L-Thalamus", 77: "R-Thalamus", 78: "L-Pallidum", 79: "R-Pallidum", 80: "Corpus-callosum", 81: "R-Lateral-ventricle-excluding-temporal-horn", 82: "L-Lateral-ventricle-excluding-temporal-horn", 83: "R-Lateral-ventricle-temporal-horn", 84: "L-Lateral-ventricle-temporal-horn", 85: "Third-ventricle", 86: "L-Precentral-gyrus", 87: "R-Precentral-gyrus", 88: "L-Straight-gyrus", 89: "R-Straight-gyrus", 90: "L-Anterior-orbital-gyrus", 91: "R-Anterior-orbital-gyrus", 92: "L-Inferior-frontal-gyrus", 93: "R-Inferior-frontal-gyrus", 94: "L-Superior-frontal-gyrus", 95: "R-Superior-frontal-gyrus", 96: "L-Postcentral-gyrus", 97: "R-Postcentral-gyrus", 98: "L-Superior-parietal-gyrus", 99: "R-Superior-parietal-gyrus", 100: "L-Lingual-gyrus", 101: "R-Lingual-gyrus", 102: "L-Cuneus", 103: "R-Cuneus", 104: "L-Medial-orbital-gyrus", 105: "R-Medial-orbital-gyrus", 106: "L-Lateral-orbital-gyrus", 107: "R-Lateral-orbital-gyrus", 108: "L-Posterior-orbital-gyrus", 109: "R-Posterior-orbital-gyrus", 110: "L-Substantia-nigra", 111: "R-Substantia-nigra", 112: "L-Subgenual-frontal-cortex", 113: "R-Subgenual-frontal-cortex", 114: "L-Subcallosal-area", 115: "R-Subcallosal-area", 116: "L-Pre-subgenual-frontal-cortex", 117: "R-Pre-subgenual-frontal-cortex", 118: "L-Superior-temporal-gyrus-anterior-part", 119: "R-Superior-temporal-gyrus-anterior-part"}
totalsegmentator_labels = {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'inferior_vena_cava', 9: 'portal_vein_and_splenic_vein', 10: 'pancreas', 11: 'adrenal_gland_right', 12: 'adrenal_gland_left', 13: 'lung_upper_lobe_left', 14: 'lung_lower_lobe_left', 15: 'lung_upper_lobe_right', 16: 'lung_middle_lobe_right', 17: 'lung_lower_lobe_right', 18: 'vertebrae_L5', 19: 'vertebrae_L4', 20: 'vertebrae_L3', 21: 'vertebrae_L2', 22: 'vertebrae_L1', 23: 'vertebrae_T12', 24: 'vertebrae_T11', 25: 'vertebrae_T10', 26: 'vertebrae_T9', 27: 'vertebrae_T8', 28: 'vertebrae_T7', 29: 'vertebrae_T6', 30: 'vertebrae_T5', 31: 'vertebrae_T4', 32: 'vertebrae_T3', 33: 'vertebrae_T2', 34: 'vertebrae_T1', 35: 'vertebrae_C7', 36: 'vertebrae_C6', 37: 'vertebrae_C5', 38: 'vertebrae_C4', 39: 'vertebrae_C3', 40: 'vertebrae_C2', 41: 'vertebrae_C1', 42: 'esophagus', 43: 'trachea', 44: 'heart_myocardium', 45: 'heart_atrium_left', 46: 'heart_ventricle_left', 47: 'heart_atrium_right', 48: 'heart_ventricle_right', 49: 'pulmonary_artery', 50: 'brain', 51: 'iliac_artery_left', 52: 'iliac_artery_right', 53: 'iliac_vena_left', 54: 'iliac_vena_right', 55: 'small_bowel', 56: 'duodenum', 57: 'colon', 58: 'rib_left_1', 59: 'rib_left_2', 60: 'rib_left_3', 61: 'rib_left_4', 62: 'rib_left_5', 63: 'rib_left_6', 64: 'rib_left_7', 65: 'rib_left_8', 66: 'rib_left_9', 67: 'rib_left_10', 68: 'rib_left_11', 69: 'rib_left_12', 70: 'rib_right_1', 71: 'rib_right_2', 72: 'rib_right_3', 73: 'rib_right_4', 74: 'rib_right_5', 75: 'rib_right_6', 76: 'rib_right_7', 77: 'rib_right_8', 78: 'rib_right_9', 79: 'rib_right_10', 80: 'rib_right_11', 81: 'rib_right_12', 82: 'humerus left', 83: 'humerus right', 84: 'scapula_left', 85: 'scapula_right', 86: 'clavicula_left', 87: 'clavicula_right', 88: 'femur left', 89: 'femur right', 90: 'hip_left', 91: 'hip_right', 92: 'sacrum', 93: 'face', 94: 'gluteus_maximus_left', 95: 'gluteus_maximus_right', 96: 'gluteus_medius_left', 97: 'gluteus_medius_right', 98: 'gluteus_minimus_left', 99: 'gluteus_minimus_right', 100: 'autochthon_left', 101: 'autochthon_right', 102: 'iliopsoas_left', 103: 'iliopsoas_right', 104: 'urinary_bladder'}
totalsegmentator2_labels = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel", 19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate", 23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1", 32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4", 41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart", 52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk", 55: "subclavian_artery_right", 56: "subclavian_artery_left", 57: "common_carotid_artery_right", 58: "common_carotid_artery_left", 59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right", 61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava", 64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right", 67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right", 71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right", 75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right", 79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right", 82: "gluteus_medius_left", 83: "gluteus_medius_right", 84: "gluteus_minimus_left", 85: "gluteus_minimus_right", 86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull", 92: "rib_right_4", 93: "rib_right_3", 94: "rib_left_1", 95: "rib_left_2", 96: "rib_left_3", 97: "rib_left_4", 98: "rib_left_5", 99: "rib_left_6", 100: "rib_left_7", 101: "rib_left_8", 102: "rib_left_9", 103: "rib_left_10", 104: "rib_left_11", 105: "rib_left_12", 106: "rib_right_1", 107: "rib_right_2", 108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8", 112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12", 116: "sternum", 117: "costal_cartilages"}
dictionaries = [moose_labels, totalsegmentator_labels, totalsegmentator2_labels]
for i, d in enumerate(dictionaries):
	dictionaries[i] = {k: v.lower() for k, v in d.items()}  # Lowercase all keys and values
	dictionaries[i].update({v: k for k, v in dictionaries[i].items()})  # Two-way dict (i.e. dict["hello"] = "world" and dict["world"] = "hello")
moose_labels, totalsegmentator_labels, totalsegmentator2_labels = dictionaries


def get_closest_matches(text_to_match, predefined_dict, n=3):
	"""Returns the n closest matching keys from the predefined dictionary to the user input."""
	# Calculate similarity for each key in the dictionary
	similarities = [(key, SequenceMatcher(None, text_to_match, key).ratio()) for key in predefined_dict.keys() if type(key) == str]
	# Sort the items based on similarity ratio in descending order and return the top n keys
	sorted_list = sorted(similarities, key=lambda x: x[1], reverse=True)
	return [item[0] for item in sorted_list[:n]]


def mask_region_num_name_converter(maskname, region_name_or_num):
	# Convert region name to num, or region num to name
	if maskname == "moose": region_labels = moose_labels
	elif maskname == "totalsegmentator" or maskname == "ts": region_labels = totalsegmentator_labels
	elif maskname == "totalsegmentator2" or maskname == "ts2": region_labels = totalsegmentator2_labels
	
	if type(region_name_or_num) == int:
		region_num = region_name_or_num
		region_name = region_labels[region_num]
	elif type(region_name_or_num) == str:
		region_name = region_name_or_num
		if region_name not in region_labels:
			suggestions = get_closest_matches(region_name, region_labels)
			print(ts.yellow + f"Mask region '{region_name}' not found in {maskname}. Did you mean one of following regions:" + ts.end)
			suggestion_choice = input("".join(["[" + str(i) + "] " + suggestion + ",  " * (i < len(suggestions) - 1) for i, suggestion in enumerate(suggestions)]) + ": ")
			try: suggestion_choice = int(suggestion_choice)
			except ValueError: raise ValueError("Suggestion choice should be an integer")
			region_name = suggestions[suggestion_choice]
		region_num = region_labels[region_name]
	else: raise TypeError("region_name_or_num should be either type int or str")
	return region_num, region_name


def load_data(*paths, load_PET_dcm_organ=None, showheader=False, silent=False):
	"""
	Load .nii.gz, .json, .tac data.
	.nii.gz will be treated as 3D or 4D PET data OR label data derived from CT.
	.json will be treated as metadata.
	.tac will be treated as IDIF data.
	Some selection will be made to .nii.gz and .json data.

	Parameters
	----------
	paths : STR / LIST of str
		Data sets, metadata. Can be in a list.

	Returns
	-------
	data : LIST
		Loaded data from given path.

	"""
	if not silent: print("Loading data...")
	if type(paths[0]) in [list, tuple]:
		paths = paths[0]
		if type(paths[0]) in [list, tuple]: raise TypeError("Can't find paths. Too many lists or tuples. Format the paths like [\"path1\",\"path2\"] or without the brackets.")
		if False:
			while type(paths[0]) == list or type(paths[0]) == tuple:
				paths = [i for i in paths[0]]
	
	for path in paths:
		if not os.path.exists(path):
			raise FileNotFoundError("Path does not exist: " + str(path))
	
	data = []
	n = 1
	for path in paths:
		if not silent: print(f"\t({n}/{len(paths)}) Loading {path}")
		n += 1
		
		if path.find(".nii.gz") != -1 or path.find(".nii") != -1:  # .nii.gz  PET/label data
			dataObject = nib.load(path)
			data.append([np.rot90(dataObject.get_fdata()), dataObject.affine])
			if showheader: print(dataObject.header)
		
		elif path.find(".npy") != -1:  # Uncompressed Numpy files
			data.append(np.load(path))
		
		elif path.find(".npz") != -1:  # Compressed Numpy files
			data.append(np.load(path)["arr_0"])
		
		elif path.find(".tac") != -1:  # .tac     IDIF
			with open(path, 'r') as f:
				data.append(np.transpose(np.loadtxt(f, delimiter='\t', skiprows=1))[2])
		
		elif path.find(".json") != -1:  # .json    metadata
			with open(path, 'r') as f:
				data.append(json.load(f))
		
		elif path.find(".ndjson") != -1:  # .ndjson
			with open(path, 'r') as f:
				data.append(ndjson.load(f))
		
		elif path.find(".txt") != -1:  # .txt
			with open(path, "r") as f:
				data.append(f.read())
		
		elif path.find(".dcm") != -1:  # Dicom file:
			dcm = pydicom.dcmread(path)
			img_arr = dcm.pixel_array
			if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
				img_arr = img_arr * dcm.RescaleSlope + dcm.RescaleIntercept
			data.append(img_arr)
		
		elif len(glob.glob(os.path.join(path, "*.dcm"))) > 0:  # Dicom directory
			patient = getdcmmeta(dcm_path=path)
			data.append(load_PET_dcm(patient.seriesuid, organ=load_PET_dcm_organ))
	
	if not silent: print("Done loading!")
	return data[0] if len(data) == 1 else data


def load_PET_dcm_worker(data):
	plane, frame, file = data
	dcm = pydicom.dcmread(file)
	img_arr = dcm.pixel_array
	if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
		img_arr = img_arr * dcm.RescaleSlope + dcm.RescaleIntercept
	del dcm
	return plane, frame, img_arr


def load_PET_dcm_worker_shm(data):  # Using shared memory
	[plane, frame, file], shm_name, sharr_shape, sharr_dtype = data
	shm = shared_memory.SharedMemory(name=shm_name)
	np_shared_array = np.ndarray(sharr_shape, dtype=sharr_dtype, buffer=shm.buf)
	dcm = pydicom.dcmread(file)
	img_arr = dcm.pixel_array
	if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
		img_arr = img_arr * dcm.RescaleSlope + dcm.RescaleIntercept
	np_shared_array[:, :, plane, frame] = img_arr
	shm.close()
	return True


def load_PET_dcm_worker_shA(infos):
	[plane, frame, file], shared_array, shared_shape = infos
	dcm = pydicom.dcmread(file)
	img_arr = dcm.pixel_array
	if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
		img_arr = img_arr * dcm.RescaleSlope + dcm.RescaleIntercept
	sharr = np.frombuffer(shared_array.get_obj()).reshape(shared_shape)
	sharr[:, :, plane, frame] = img_arr


def load_PET_dcm(PET_seriesuid, organ=None, planes=None, indexing=0, start="reversed"):  # Only for loading dynamic PET scans
	"""
	Load dynamic PET DICOM files as Numpy array. Load only the planes of a given totalsegmentator organ name (or label number, not yet), or provide min and max plane indices.
	To get metadata use "dcm2niix -b o ..." to output only the meta file.
	DICOM planes are one-indexed starting from top of patient.
	Either a totalsegmentator organ can be given, which uses the .json file in PET derivatives directory to find the planes.
	Planes can also manually be given as a tuple of min and max plane: (min_plane, max_plane). Planes uses 1-indexing starting from the top of the patient, however if planes are given, this can be changed using [indexing=1] (default:0) and [start="reversed"] (default:"reversed").

	Parameters
	----------
	PET_seriesuid
	organ:  totalsegmentator organ name
	planes:  tuple-like, (min plane index, max plane index), both included. One-indexed starting from top of patient.
	indexing: Only applies if planes is given

	Returns
	-------
	4D numpy array: [coronal, sagittal, axial, frame]
	Tuple of min and max plane for use with mask, 1-indexed and starting at "start": (min_plane, max_plane)
	"""
	if psutil.cpu_percent(0.2) / 100 >= 0.88:
		use_mp = 0
	
	if len(glob.glob(f"/raid/source/*/*/{PET_seriesuid}.tsv")) > 0:
		PET_info = pd.read_csv(glob.glob(f"/raid/source/*/*/{PET_seriesuid}.tsv")[0], sep='\t')
	
	elif len(glob.glob(f"/raid/source/*/*/{PET_seriesuid}.json")) > 0:
		PET_info = load_data(glob.glob(f"/raid/source/*/*/{PET_seriesuid}.json")[0])
		for _ in range(2):  # The first two entries (StudyUID and SeriesUID) are not needed. We want the underlying dict.
			PET_info = PET_info[next(iter(PET_info))]
		
		dcm_filelist = [{key: value[i] for i, key in enumerate(PET_info["FilelistHeader"])} for value in PET_info["Filelist"]]
		PET_info = pd.DataFrame.from_dict(dcm_filelist)
	
	# PET_info["Plane"] = PET_info["Plane"]  # One-indexed
	# PET_info["Frame"] = PET_info["Frame"]
	N_planes = max(PET_info["Plane"]) - min(PET_info["Plane"]) + 1
	N_frames = max(PET_info["Frame"]) - min(PET_info["Frame"]) + 1
	
	if start == "reversed":
		PET_info["Plane"] = N_planes + 1 - PET_info["Plane"]
	
	min_plane_organ, max_plane_organ, min_plane_plane, max_plane_plane = [None for _ in range(4)]
	
	if organ:
		# Get planes containing organ
		if len(glob.glob(f"/raid/derivatives/*/*/{PET_seriesuid}/totalsegmentator*.json")) == 0:
			raise FileNotFoundError("totalsegmentator.json file could not be found")
		
		with open(glob.glob(f"/raid/derivatives/*/*/{PET_seriesuid}/totalsegmentator*.json")[0], 'r') as f:
			totalsegmentator_json = json.load(f)
		
		totalsegmentator_json[organ]["Plane"] = np.asarray(totalsegmentator_json[organ]["Plane"])
		# TODO: use totalsegmentator organ name OR label id
		if start == "reversed":
			totalsegmentator_json[organ]["Plane"] = N_planes - totalsegmentator_json[organ]["Plane"]
		
		organ_planes = totalsegmentator_json[organ]["Plane"]
		min_plane_organ, max_plane_organ = min(organ_planes), max(organ_planes)  # Zero indexed (at least should be!)
	
	if planes:
		min_plane_plane, max_plane_plane = [x + (1 - indexing) for x in planes]  # Converting planes from "indexing"-indexing to 1-indexing
		if max_plane_plane < 1:
			max_plane_plane += N_planes
		if min_plane_plane < 1:
			min_plane_plane += N_planes
		
		if max_plane_plane < min_plane_plane:   raise IndexError("Min-plane can not be greater than max-plane")
		if min_plane_plane < 1:                 raise IndexError(f"Min-plane can not be less than {indexing}")
		if max_plane_plane > N_planes:          raise IndexError(f"Max-plane can not be greater than the number of planes ({N_planes})")
	
	if not organ and not planes:  # If neither are given, load the whole scan
		min_plane_plane, max_plane_plane = 1, N_planes
	
	# If both are given, take the min & max of either
	min_plane, max_plane = min([x for x in [min_plane_organ, min_plane_plane] if x is not None]), max([x for x in [max_plane_organ, max_plane_plane] if x is not None])
	num_planes = max_plane - min_plane + 1
	# print("min_plane, max_plane:", f"{min_plane}-{max_plane} = {max_plane - min_plane} ({num_planes} planes)")
	PET_info_planes = PET_info[PET_info["Plane"] >= min_plane][PET_info["Plane"] <= max_plane]
	first_dcm = pydicom.dcmread(PET_info_planes["File"].iloc[0])
	img_height, img_width = first_dcm.pixel_array.shape
	print("Allocating memory...")
	dcm_loaded_arr = np.empty((img_height, img_width, num_planes, N_frames))  # Allocating in virtual memory
	dcm_loaded_arr[:, :, :, :] = 0.0  # Allocating in physical RAM
	for plane, frame, file in tqdm(zip(PET_info_planes["Plane"] - min_plane, PET_info_planes["Frame"] - min(PET_info_planes["Frame"]), PET_info_planes["File"]), desc="Loading PET dcm (no mp)", ncols=100, unit_scale=True, smoothing=0.01, total=num_planes * N_frames):
		dcm = pydicom.dcmread(file)
		img_arr = dcm.pixel_array
		if 'RescaleSlope' in dcm and 'RescaleIntercept' in dcm:
			img_arr = img_arr * dcm.RescaleSlope + dcm.RescaleIntercept
		dcm_loaded_arr[:, :, plane, frame] = img_arr
	
	return dcm_loaded_arr, (min_plane, max_plane)


def savenii(img, affine, filepath):
	img = nib.Nifti1Image(np.rot90(img, -1), affine)
	nib.save(img, filepath)


def dynamic_multiprocessing_adaptor(input_queue, output_queue, worker, chunks_per_worker=1, *args_to_pass):
	try:
		for _ in range(chunks_per_worker):
			queue_data = input_queue.get()
			if queue_data is None:
				break
			time0 = time()
			process_num, data = queue_data
			results = worker(data, *args_to_pass)  # , DYNAMIC_PROCESSING_process_num=process_num)  # TODO: The process num could maybe be passed to the worker
			output_queue.put((process_num, time() - time0, results))
	except Exception as e:
		print(ts.red, e, ts.end)
		raise e


def dynamic_multiprocessing(data, worker, *args_to_pass, auto=False, chunks_per_worker=1, min_processes=1, max_processes=multiprocessing.cpu_count() - 3, target_load=0.90, interval=0.2, show_stats=False, show_ETC=True, use_threading=False):
	"""
	Dynamically changing the number of processes to use for multiprocessing depending on the total CPU load of the system.
	If the system CPU load changes (e.g. from another script running), this will stop the spawning of new processes until the CPU load is below target_load.
	One new processes will be spawned every interval seconds if the CPU load is below target_load.
	Each process will run the worker function on chunks_per_worker number of entries in data.
	The time it takes this to respond to increased CPU load depends on the calculation time of a single data entry and the number of chunks_per_worker before a process is killed.
	Similarly, the time it takes to increase the number of processes from 0, means that the interval should be lower than the time of a process (calculation time of one entry * chunks_per_worker).
	Therefore, choose chunks_per_worker such that it is about 100 times the  [...] # TODO: give a formula. Maybe it's even possible to do this all automatically by timing one process and changing interval and chunks_per_worker accordingly (maybe even dynamically :D).

	Parameters
	----------
	data:               The data in list form that need processing by worker.
	worker:             The function that processes a single entry of data at a time.
	args_to_pass:       Any arguments, constant across data, that need to be passed to the worker function.
	auto:               Set chunks_per_worker automatically from the time it takes a single process to execute. Interval and chunks_per_worker are set to default initially.
	chunks_per_worker:  Number of data entries to be taken by a process before closing it.
	min_processes:      Minimum number of processes to spawn regardless of CPU load. Minimum is 1.
	max_processes:      Limit the maximum number of simultaneous processes. Can max be the number of CPU available on the system.
	target_load:        Spawn processes to keep the CPU at this level (0 to 1; default: 0.9).
	interval:           The interval (in seconds) used to check for CPU load. Hence, also the time between spawning of new processes. Min 0.1.
	show_stats:         Show the CPU usage and the number of processes used for every interval.

	Returns
	-------
	Yields the return value of worker in the same order as data.
	"""
	# show_stats = True
	if auto:
		interval = 0.2
		chunks_per_worker = 1
	# interval_dyn = interval
	iteration_time = 0
	
	print(f"\rPreparing dynamic multi{'processing' * (not use_threading) + 'threading' * use_threading}...", end="\r")
	
	input_queue = multiprocessing.Queue()
	output_queue = multiprocessing.Queue()
	
	for xi, x in enumerate(data):
		input_queue.put((xi, x))
	
	for _ in range(multiprocessing.cpu_count()):  # Put extra Nones in the input_queue to stop the workers when reaching the end
		input_queue.put(None)
	
	results_list = [None] * len(data)
	
	args = [input_queue, output_queue, worker, chunks_per_worker] + list(args_to_pass)
	
	max_processes = min(max_processes, multiprocessing.cpu_count() - 3)  # Total number of CPUs to use (can maybe be more???)
	min_processes = max(1, min(min_processes, max_processes))
	current_processes = []
	num_results_received = 0
	next_result_i_to_yield = 0
	
	rate = 1  # s/it
	alpha_rate = 0.0001  # Adjust how fast to adapt to new rate
	time_to_start_new_process = 1
	alpha_process = 0.1
	cpw_multiplier = 3  # multiplier for auto-adjusting chunks_per_worker. Determines how many times (x * target_load * total_num_cpu) a process runs before stopping and being replaced. Should be between 1.5 and ~4. Low value makes it respond faster to CPU load, but can be too short to have time for new processes to be spawned and keep CPU at target_load; high value lowers the number of process starts and hence overhead of starting new processes.
	process_alive_duration = 1.5 * target_load * multiprocessing.cpu_count()  # s. Max duration that a process will be kept alive by adjusting chunks_per_worker. Also determines how fast it will respond to a surge in CPU load.
	alpha_iteration_time = 0.9  # For new iteration_time > prev iteration_time. (1 - alpha_iteration_time) otherwise.
	time_start = time()
	ETC_total_num = len(data)
	ETC_T = 1
	alpha_ETC_T = 0.1
	ETC_T_time = time_start
	prev_num_results_received = 0
	ETC_Q_T = 1
	ETC_Q_Ts = [ETC_Q_T] * int(1.5 * 60 / interval)  # saving time points for 1.5 minutes
	ETC_Q_times = [time_start] * int(1.5 * 60 / interval)  # saving time points for 1.5 minutes
	ETC_Q_nums_results = [0] * int(1.5 * 60 / interval)  # saving time points for 1.5 minutes
	ETC_Q_counter = 0
	alpha_ETC_Q_T = 0.5
	ETC_Q_T_mean_diff = 0
	try:
		while num_results_received < len(results_list):
			time_while = time()
			cpu_load = psutil.cpu_percent(interval=max(interval, 0.1)) / 100
			if show_stats: print(" "
			                     f"CPU: {ts.yellow * (cpu_load < target_load) + ts.green * (target_load <= cpu_load) + ts.red * (cpu_load >= (1 - 1.5 / multiprocessing.cpu_count()))}{roundsig(cpu_load * 100, 3)}%{ts.end},  "
			                     f"procs: {ts.yellow * (len(current_processes) < min_processes or len(current_processes) >= max_processes)}{len(current_processes)}{ts.end},  "
			                     f"rate: {roundsig(rate, 3)}s/it,  "
			                     f"cpw: {chunks_per_worker},  "
			                     f"tnp: {roundsig(time_to_start_new_process, 3)}s,  "
			                     f"prt: {roundsig(rate * chunks_per_worker, 3)}s,  "
			                     f"itt: {roundsig(iteration_time, 3)}s  ")
			
			# Check for results
			while not output_queue.empty():
				# print(1, end="")
				xi, ti, x = output_queue.get()
				results_list[xi] = x
				num_results_received += 1
				if auto:  # and time() - time0 < 200:
					rate = alpha_rate * ti + (1 - alpha_rate) * rate if num_results_received > 1 else ti
			# rate = (rate * (num_results_received - 1) + ti) / num_results_received  # Average rate
			# rate.append(ti)  # s/it
			
			while next_result_i_to_yield < len(results_list) and results_list[next_result_i_to_yield] is not None:
				# print(2, end="")
				yield results_list[next_result_i_to_yield]
				next_result_i_to_yield += 1
			
			# Clean up finished processes
			current_processes = [p for p in current_processes if p.is_alive()]
			
			# Start new process
			if not input_queue.empty():
				if len(current_processes) < max_processes and (cpu_load < target_load or len(current_processes) < min_processes):
					# Start a new worker
					time1 = time()
					if not use_threading:
						p = multiprocessing.Process(target=dynamic_multiprocessing_adaptor, args=args)
					elif use_threading:
						p = threading.Thread(target=dynamic_multiprocessing_adaptor, args=args)
					p.start()
					current_processes.append(p)
					time_to_start_new_process = alpha_process * (time() - time1) + (1 - alpha_process) * time_to_start_new_process if len(current_processes) > 1 else time() - time1
			
			if show_ETC and num_results_received > 0:
				# if prev_num_results_received < num_results_received:
				# 	ETC_T = alpha_ETC_T * (time() - ETC_T_time) / (num_results_received - prev_num_results_received) * (ETC_total_num - num_results_received) + (1 - alpha_ETC_T) * ETC_T if type(ETC_T) != int else (time() - time_start) / max(1, num_results_received) * (ETC_total_num - num_results_received)
				# 	ETC = datetime.now() + timedelta(seconds=ETC_T) + timedelta(minutes=-5)  # 5 min bias from Mogensen server time to PC time
				# 	ETC_T_time = time()
				# 	prev_num_results_received = num_results_received
				
				ETC_Q_times[ETC_Q_counter] = time()
				ETC_Q_nums_results[ETC_Q_counter] = num_results_received
				# ETC_Q_T = alpha_ETC_Q_T * (time() - ETC_Q_times[(ETC_Q_counter + 1)%len(ETC_Q_times)]) / (num_results_received - ETC_Q_nums_results[(ETC_Q_counter + 1)%len(ETC_Q_nums_results)]) * (ETC_total_num - num_results_received) + (1 - alpha_ETC_Q_T) * ETC_Q_T if type(ETC_Q_T) != int else rate * ETC_total_num  # EMA
				# ETC_Q_T = (time() - ETC_Q_times[(ETC_Q_counter + 1)%len(ETC_Q_times)]) / (num_results_received - ETC_Q_nums_results[(ETC_Q_counter + 1)%len(ETC_Q_nums_results)]) * (ETC_total_num - num_results_received)  # Straight average
				# if type(ETC_Q_Ts[0]) == int:
				# 	ETC_Q_Ts = [ETC_Q_T] * len(ETC_Q_Ts)
				# ETC_Q_Ts[ETC_Q_counter] = ETC_Q_T
				# ETC_Q_T_mean_diff = ETC_Q_T - np.mean(ETC_Q_Ts)
				# ETC_Q = datetime.now() + timedelta(seconds=ETC_Q_T) + timedelta(minutes=-5)  # 5 min bias from Mogensen server time to PC time
				ETC_Q_counter = (ETC_Q_counter + 1) % len(ETC_Q_nums_results)
				
				ETC_T = alpha_ETC_T * np.mean([(time() - ETC_Q_times[i]) / (num_results_received - ETC_Q_nums_results[i]) * (ETC_total_num - num_results_received) for i in range(len(ETC_Q_times)) if ETC_Q_nums_results[i] < num_results_received]) + (1 - alpha_ETC_T) * ETC_T if type(ETC_T) != int else np.mean([(time() - ETC_Q_times[i]) / (num_results_received - ETC_Q_nums_results[i]) * (ETC_total_num - num_results_received) for i in range(len(ETC_Q_times)) if ETC_Q_nums_results[i] < num_results_received])  # Straight average
				ETC = datetime.now() + timedelta(seconds=ETC_T) + timedelta(minutes=-6.3)  # 5 min bias from Mogensen server time to PC time
				
				ETC_avg = datetime.now() + timedelta(seconds=(time() - time_start) / (num_results_received - 0) * (ETC_total_num - num_results_received)) + timedelta(minutes=-6.3)  # Average ETC
				
				def frelative_day(ETC):  # TODO: Maybe move this to beginning of dyn_multiproc
					relative_day = ""  # if today
					if ETC.date() == datetime.now().date() + timedelta(days=1): relative_day = "Tomorrow"
					if ETC.date() > datetime.now().date() + timedelta(days=1):  relative_day = ETC.strftime("%a")  # "Mon" if ETC.isoweekday() == 1 else "Tue" if ETC.isoweekday() == 2 else "Wed" if ETC.isoweekday() == 3 else "Thu" if ETC.isoweekday() == 4 else "Fri" if ETC.isoweekday() == 5 else "Sat" if ETC.isoweekday() == 6 else "Sun" if ETC.isoweekday() == 7 else ""
					if ETC.date() >= datetime.now().date() + timedelta(days=7): relative_day = ETC.strftime("%d %b")
					return relative_day
				
				print(" --> " +
				      f"{frelative_day(ETC) + ' ' * bool(frelative_day(ETC))}{ETC.strftime('%H:%M')}".ljust(len("Tomorrow") + 1 + 5) + " | " +
				      # f"{frelative_day(ETC_Q) + ' '*bool(frelative_day(ETC_Q))}{ETC_Q.strftime('%H:%M')} {'-' if np.sign(ETC_Q_T_mean_diff) == -1 else '+'} {min(999, int(abs(ETC_Q_T_mean_diff) / 60))}".ljust(len("Tomorrow") + 1 + 5 + 3 + 3) + " | " +
				      f"{frelative_day(ETC_avg) + ' ' * bool(frelative_day(ETC_avg))}{ETC_avg.strftime('%H:%M')}".ljust(len("Tomorrow") + 1 + 5), end="\r")
			
			# Adjust chunks_per_worker
			if auto:  # and time() - time0 < 200:# and not chunks_per_worker >= 2 * interval * (target_load * multiprocessing.cpu_count()) / rate:  # Assuming steady-state after 2 mins
				chunks_per_worker = max(1, min(int(process_alive_duration / rate), int(cpw_multiplier * (interval + time_to_start_new_process) * target_load * multiprocessing.cpu_count() / rate)))  # Multiplied by 2 to have a check twice as often for faster
				args[3] = chunks_per_worker
			
			iteration_time = 0.75 * (time() - time_while) + (1 - 0.75) * iteration_time if 0.9 * iteration_time < (time() - time_while) < 1.5 * iteration_time else 0.01 * (time() - time_while) + (1 - 0.01) * iteration_time
			sleep(max(0, iteration_time - (time() - time_while)))  # To keep the same duration for each while-iteration
	# interval_dyn_time = max(0, interval + time_to_start_new_process - (time() - time0))
	# interval_dyn = interval + max(0, interval + time_to_start_new_process - (time() - time0))  # To keep the same duration for each while-iteration
	
	except Exception as e:
		print(e)
		print("Terminating processes...")
		for p in current_processes:
			p.terminate()
		current_processes = [p for p in current_processes if p.is_alive()]
	
	finally:
		for p in current_processes:
			p.join()


def roundsig(n, s, returnasstring=False):
	# Round a number, n, to s significant digits.
	# if returnasstring:
	#     # m = f"{0:.{s}g}".format(n)
	#     m = f"{n:.{s}f}"
	# else:
	try:
		if n == 0:
			m = n
		else:
			m = round(n, -int(np.floor(np.log10(abs(n)))) + (s - 1))
	except Exception as e:
		m = n
		print("roundsig ERROR:", str(e))
	if returnasstring:
		m = str(m)
		if m[-2:] == ".0":
			m = m[:-2]
	return m


class getdcmmeta:
	def __init__(self, seriesuid=None, dcm_path=None, printheader=False):
		if seriesuid is not None and dcm_path is None:
			dcmfile = next(glob.iglob("/raid/source/*/*/" + seriesuid + "/*.dcm"))
		elif seriesuid is None and dcm_path is not None:
			dcmfile = next(glob.iglob(dcm_path + "/*.dcm"))
		self.header = pydicom.dcmread(dcmfile, stop_before_pixels="only_header")
		if printheader: print(self.header)
		
		self.cpr = self.header[0x10, 0x20].value
		self.date = self.header[0x8, 0x20].value
		self.studyuid = self.header[0x20, 0xd].value
		self.seriesuid = self.header[0x20, 0xe].value
		self.acct_studyuid = self.header[0x8, 0x1250].value[0][0x20, 0xd].value
		self.acct_seriesuid = self.header[0x8, 0x1250].value[0][0x20, 0xe].value
		self.study_description = self.header[0x8, 0x1030].value
		self.series_description = self.header[0x8, 0x103e].value
		self.tracer = self.header[0x54, 0x16].value[0][0x18, 0x31].value.lower()
		self.sdir = os.path.join("/raid/source", self.cpr, self.studyuid, self.seriesuid)
		self.ddir = os.path.join("/raid/derivatives", self.cpr, self.studyuid, self.seriesuid)
		self.acctdir = os.path.join("/raid/derivatives", self.cpr, self.acct_studyuid, self.acct_seriesuid)


# %% Edge detection from labels
def edge(labels):
	labels = labels > 0
	coronal_mask = np.zeros(labels.shape)
	# First and last indices along coronal axis
	first = np.argmax(labels, axis=0)
	last = labels.shape[0] - np.argmax(labels[::-1, ...], axis=0)
	for sagittal, axial in tqdm([[y, z] for y in range(labels.shape[1]) for z in range(labels.shape[2])], desc="Coronal", ncols=100):
		if not np.all(labels[:, sagittal, axial] == 0):
			coronal_mask[first[sagittal, axial]:last[sagittal, axial], sagittal, axial] = 1
	
	sagittal_mask = np.zeros(labels.shape)
	# First and last indices along sagittal axis
	first = np.argmax(labels, axis=1)
	last = labels.shape[1] - np.argmax(labels[:, ::-1, :], axis=1)
	for coronal, axial in tqdm([[x, z] for x in range(labels.shape[0]) for z in range(labels.shape[2])], desc="Sagittal", ncols=100):
		if not np.all(labels[coronal, :, axial] == 0):
			sagittal_mask[coronal, first[coronal, axial]:last[coronal, axial], axial] = 1
	
	axial_mask = np.zeros(labels.shape)
	# First and last indices along axial axis
	first = np.argmax(labels, axis=2)
	last = labels.shape[2] - np.argmax(labels[..., ::-1], axis=2)
	for coronal, sagittal in tqdm([[x, y] for x in range(labels.shape[0]) for y in range(labels.shape[1])], desc="Axial", ncols=100):
		if not np.all(labels[coronal, sagittal, :] == 0):
			axial_mask[coronal, sagittal, first[coronal, sagittal]:last[coronal, sagittal]] = 1
	
	return coronal_mask * sagittal_mask * axial_mask


# %% Nearest neighbor
def NN_mp(data, weights):
	return ndimage.convolve(data, weights, mode="constant", cval=0)


def nearest_neighbor(data, weights=None, p_cpu=N_CPUs_to_use()):  # max(1, int(multiprocessing.cpu_count()/2))):
	"""
    Input 4D data (x, y, z, t).
    (3D data should also be possible - function needs modification if it's ever needed.)

    Parameters
    ----------
    data : 4D-ARRAY
        .
    weights : 4D-ARRAY (same shape as data), optional
        Weights for averaging. The default is a 3^3 cube with inverse length.
    p_cpu

    Returns
    -------
    4D-ARRAY
        .

    """
	if weights is None or weights == "3-space-euclidian":  # Standard 3*3*3 cube in space weighted by euclidian distance
		weights = np.array([[[[1 / (i ** 2 + j ** 2 + k ** 2 + 1)] for i in range(-1, 2)] for j in range(-1, 2)] for k in range(-1, 2)])
	elif weights == "5-space-euclidian":  # Same as above but with a cube of 5*5*5
		weights = np.array([[[[1 / (i ** 2 + j ** 2 + k ** 2 + 1)] for i in range(-2, 3)] for j in range(-2, 3)] for k in range(-2, 3)])
	elif weights == "3-space-equal":  # 3*3*3 cube, equal weights
		weights = np.ones((3, 3, 3))
	elif weights == "5-space-equal":  # 5*5*5 cube, equal weights
		weights = np.ones((5, 5, 5))
	elif weights == "7-space-equal":  # 7*7*7 cube, equal weights
		weights = np.ones((7, 7, 7))
	
	weights = weights / np.sum(weights)
	# return ndimage.convolve(data, weights, mode="constant", cval=0)
	
	# p_cpu = 30
	data_mp = [data[:, :, :, i] for i in range(data.shape[-1])]
	# data_NN = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=p_cpu) as executor:
		worker = partial(NN_mp, weights=weights)
		results = list(tqdm(executor.map(worker, data_mp), desc="NN", total=len(data_mp)))
	
	# for result in results:
	#     data_NN.append(result)
	#     print(np.shape(result), "\t", np.shape(data_NN))
	
	print("Nearest neighbors done! Stacking...")
	return np.stack(results, axis=-1)


def NN5_low_memory_worker(TAC_NN5, meta, input_function):
	return create_delaymap(TAC_NN5, meta, input_function, return_errormap=True, return_gofmap=True, return_poptmap=True, save_temp=False, use_temp=False, afterburner=False, is_afterburner=True)


def bendylinfit(x, a, b, c, d=0):  # , d):  # TODO: removed d parameter
	return a * b * np.log(np.exp((c - x) / b) + 1) + a * (x - c)  # + d


def tracer_appearance(PET_list, meta, return_popt=False, verbose=False, **kwargs):
	if "FrameReferenceTime" in meta.keys():
		FrameReferenceTime = meta["FrameReferenceTime"]  # TODO: FrameReferenceTime not used in dcm2niix anymore. Calculate it manually.
	elif "FrameTimesStart" in meta.keys() and "FrameDuration" in meta.keys():
		FrameReferenceTime = list(np.array(meta["FrameTimesStart"]) + 0.5 * np.array(meta["FrameDuration"]))
	else:
		raise KeyError("'FrameReferenceTime', 'FrameTimesStart', 'FrameDuration' keys not in meta. Cannot calculate find the frame times")
	sloperef = np.array([(1) / (FrameReferenceTime[i + 1] - FrameReferenceTime[i]) for i in range(len(FrameReferenceTime) - 1)]);
	sloperef = np.append(sloperef[0], sloperef)
	FrameDuration = meta["FrameDuration"]
	FrameTimesStart = meta["FrameTimesStart"]
	
	def lefit(dat, meta, fitStopIndex):
		try:
			x = list(np.arange(-10, 0)) + FrameReferenceTime[:fitStopIndex + 1]
			y = [0] * 10 + list(dat[:fitStopIndex + 1])
			popt, pcov = curve_fit(bendylinfit, x, y, p0=[6e4, 2, 30], bounds=((0.1, 0.1, 0.1), (2e7, 7, 120)))  # , sigma=[0.2] * 10 + [1] * (len(x) - 10))#, method="dogbox")#, bounds=((0.1, 0.01, 0, 0), (2e7, 10, 120, 1)))#, max_nfev=4000, sigma=[0.2] * 10 + [1] * (len(x) - 10))  # , p0=[40e3,0.5,20,1e3,0.5,30], bounds=((500,0.1,5,0,1e-3,10), (80e3,20,45,20e3,15,100))) #p0=[40e3,0.5,20], bounds=((500,0.1,5), (80e3,20,40)))  # TODO: p0 (c): 18, bound-upper (c): 60, maxfev: 800, maxfev:4000, sigma:[0.2] * 10 + [1] * (len(x) - 10), bound-upper (d): 1e5, (b): 10
		except Exception as e:
			if verbose: print("FITTING ERROR:", e)
			popt = [-1, -1, -1]  # , -1]
			pcov = [0, 0, 0]  # , 0]
		return popt, pcov
	
	n, N, rate = 0, 1, 1 / 10  # Just something random to not make them undefined... Planning to remove them entirely
	if type(PET_list) != list: PET_list = [PET_list]  # If there's just one TAC in it, and I forgot to put it in a list... # Todo: probably make this better. Works ok for now, but what if a list is being passed and not a numpy array? Right now I'm taking a list of numpy arrays.
	
	PET_cumsum = np.cumsum(PET_list / sloperef, axis=1)
	
	time1 = time()
	TOI = []
	EOI = []
	popt = []
	perr = []
	gof = []
	searchrange = []
	for i in tqdm(range(len(PET_cumsum)), desc=f"{n}", disable=True):  # =len(PET_list) == 1):
		if verbose: print()  # Just print a newline, so the next doesn't start on the end of the tqdm
		TOI.append([])
		EOI.append([])
		popt.append([])
		perr.append([])
		gof.append([])
		thegoodfits = []  # Just set to something, will be overwritten later in the code, but not if there are no peaks found
		
		searchrange.append([0, 120])
		
		# Plateau detection
		plateau_min_length = 3  # The minimum number of points to be considered a plateau [points]
		plateau_max_time_between = 12  # Max time between plateaus [s]
		plateau_threshold = 0.001 * max(PET_cumsum[i][:np.argmax(np.asarray(FrameReferenceTime) > 120)])  # Threshold for the plateau to be within
		plateau = [0]
		for k in range(0, min(len(PET_cumsum[i]) - plateau_min_length, np.argmax(PET_cumsum[i] > PET_cumsum[i][-1] / 4), np.argmax(np.asarray(FrameReferenceTime) > 120))):  # Searches up to the smallest of either the length of the TAC, 1/4 of the max cumsum value (in case it flattens out quickly), or the first two mintes
			if PET_cumsum[i][k + plateau_min_length] - PET_cumsum[i][k] < plateau_threshold and FrameReferenceTime[k] - FrameReferenceTime[plateau[-1]] <= plateau_max_time_between:  # last term makes sure two plateaus are not too far apart in seconds
				plateau.append(k)
		if len(plateau) == 0:
			plateau.append(0)
		
		if verbose: print("Plateaus at:", plateau, "within", roundsig(plateau_threshold, 3), "Bq/mL")
		PET_cumsum[i] = np.asarray([0] * plateau[-1] + list(np.asarray(PET_cumsum[i][plateau[-1]:]) - PET_cumsum[i][plateau[-1]]))
		PET_list[i] = np.asarray([0] * plateau[-1] + list(PET_list[i][plateau[-1]:]))
		
		# Fit stop indices using acceleration of cumsum
		metadiff1 = np.asarray(FrameReferenceTime[2:]) - np.asarray(FrameReferenceTime[1:-1])
		metadiff2 = np.asarray(FrameReferenceTime[1:-1]) - np.asarray(FrameReferenceTime[:-2])
		data_acc = ((PET_cumsum[i][2:] - PET_cumsum[i][1:-1]) / metadiff1 - (PET_cumsum[i][1:-1] - PET_cumsum[i][:-2]) / metadiff2) / (metadiff1 + metadiff2)
		
		fitStopIndex, peak_prop = signal.find_peaks(np.abs(-data_acc[:np.argmax(np.asarray(FrameReferenceTime) > 120)]), height=0)  # , distance=2)  # TODO: height=150; height=125; height=20; distance=2
		peak_heights = peak_prop["peak_heights"]
		fitStopIndex = fitStopIndex[np.argsort(peak_heights)[::-1]] + 1  # +1 because acc starts at index 1
		peak_heights = peak_heights[np.argsort(peak_heights)[::-1]]
		fitStopIndex, peak_heights = [x[fitStopIndex <= np.argmax(np.asarray(FrameReferenceTime) >= 120)] for x in [fitStopIndex, peak_heights]]  # Taking only the first two minutes
		
		# Fit stop indices using 3-point circle to find curvature
		
		if verbose: print("fitStopIndices:", fitStopIndex)  # TODO: The first fitStopIndex seems to always be 39 (i.e. where frame duration changes to 2 sec)
		
		for j in range(min(10, len(fitStopIndex))):
			a, b = lefit(PET_cumsum[i], meta, fitStopIndex[j])
			popt[i].append(a)
			perr[i].append(np.sqrt(np.diag(b)))
			
			if popt[i][j][0] == -1 or perr[i][j][0] == np.inf:  # Error: Bad fit
				TOI[i].append(-1)
				EOI[i].append(-1)
				gof[i].append(-1)
			else:  # TOI IS CALCULATED HERE !!!
				TOI[i].append(popt[i][j][2] - 3.5 * popt[i][j][1])  # - popt[0]*popt[1]/3
				EOI[i].append(np.sqrt(perr[i][j][2] ** 2 + 3.5 * perr[i][j][1] ** 2))  # Not totally correct propagation of errors
				# Goodness of fit (R^2)
				number_of_extra_points = 2
				SSres = np.sum([(PET_cumsum[i][k] - bendylinfit(FrameReferenceTime[k], *popt[i][j])) ** 2 for k in range(min(fitStopIndex[j] + number_of_extra_points, len(FrameReferenceTime)))])
				SStot = np.sum([(PET_cumsum[i][k] - np.mean(PET_cumsum[i][:fitStopIndex[j] + 1 + number_of_extra_points])) ** 2 for k in range(min(fitStopIndex[j] + number_of_extra_points, len(FrameReferenceTime)))])
				R_2 = 1 - SSres / SStot
				gof[i].append(R_2)
				if TOI[i][j] < 0:  # Error: Time less than 0
					TOI[i][j] = -2
					EOI[i][j] = -2
					gof[i][j] = -2
			
			if verbose: print(f"fitStopIndex: {fitStopIndex[j]}"
			                  f" ({FrameReferenceTime[fitStopIndex[j]]} s)"
			                  f" ({roundsig(peak_heights[j], 3)}):"
			                  f" \t{[roundsig(x, 4) for x in [TOI[i][j], EOI[i][j], gof[i][j]]]}"
			                  f" \tpopt: {[roundsig(x, 4) for x in popt[i][j]]}")
		
		thegoodfits = np.argsort(gof[i])[::-1]
		thegoodfits = thegoodfits[np.asarray(gof[i])[thegoodfits] >= 0.8]
		thegoodfits = thegoodfits[0 <= np.asarray(TOI[i])[thegoodfits]]  # 0 <= T
		thegoodfits = thegoodfits[np.asarray(TOI[i])[thegoodfits] < np.asarray(FrameReferenceTime)[fitStopIndex[thegoodfits]]]  # T < fitStopIndex time
		thegoodfits = thegoodfits[np.asarray(EOI[i])[thegoodfits] <= np.median(np.asarray(EOI[i])[thegoodfits])]
		
		if verbose: print(f"fit[{thegoodfits}]:", np.asarray(fitStopIndex)[thegoodfits], f"({np.asarray(FrameReferenceTime)[np.asarray(fitStopIndex)[thegoodfits]]} s)")
		
		try:
			if len(thegoodfits) > 0:
				popt[i] = np.median(np.asarray(popt[i])[thegoodfits], axis=0)
				TOI[i] = popt[i][2] - 3.5 * popt[i][1]
				EOI[i] = np.median(np.asarray(EOI[i])[thegoodfits])
				gof[i] = np.median(np.asarray(gof[i])[thegoodfits])
			else:
				popt[i] = [-1, -1, -1]
				TOI[i] = -1
				EOI[i] = -1
				gof[i] = -1
		except Exception as e:
			print("There was an error:", e)
			popt[i] = [-1, -1, -1]
			TOI[i] = -1
			EOI[i] = -1
			gof[i] = -1
		
		if verbose:  print("Final:", [roundsig(x, 5) for x in [TOI[i], EOI[i], gof[i]]], [roundsig(x, 5) for x in popt[i]])
	
	rate = (time() - time1) / len(PET_cumsum)
	if not return_popt: everything = [[n, N, rate]] + [[TOI[i], EOI[i], gof[i], [popt[i][0], popt[i][1]]] for i in range(len(TOI))]  # Extra "n" to account for chunk number
	if return_popt: everything = [[n, N, rate]] + [[TOI[i], EOI[i], gof[i], popt[i]] for i in range(len(TOI))]  # Returns full popt instead of only a and b. Should only be used for small datasets! Extra "n" to account for chunk number
	return everything  # "everything" is the full packet of [bunchinfo] + [results]


def create_delaymap(PET, meta, input_function, mask=None, N_cpu=multiprocessing.cpu_count(), NN5_low_memory=True, return_errormap=False, return_gofmap=False, return_poptmap=False, save_temp=True, use_temp=False, temp_folder=None, afterburner=True, is_afterburner=False, verbose=False):
	"""
	Create delay map from 4D array and input function.
	If PET is a single TAC (1D Numpy array), only PET, meta and input_function need to be provided, and it will return the estimated time. If delay could not be estimated, it will return Numpy.nan.
	Parameters
	----------
	PET:            4D Numpy array
	meta:           dict contains frame time information (meta["FrameReferenceTime"]: list; meta["FrameDuration"]: list; meta["FrameTimesStart"]: list)
	input_function: input function with which calculate the difference in onset time, i.e. the delay.
	mask: 3D Numpy array to mask the PET array.
	N_cpu: number of CPUs to use for multiprocessing.
	return_errormap: return the onset time error calculated from the fit parameter errors.
	return_gofmap: return a map of the goodness of fit (R^2).
	return_poptmap: return a map of the optimal fit parameters a and b.
	save_temp: save the result batches in case the connection is lost/script stops/program crashes.
	use_temp: use the previously saved temporary result file.
	temp_folder: specify the location of the temporary result file.
	afterburner: rerun create_delaymap() with the voxels with errors.
	is_afterburner: whether the current run is the afterburner or not.

	Returns
	-------
	delaymap: 3D Numpy array. Values in seconds.
	"""
	
	# If PET is a single TAC, just return a single delay
	if PET.ndim == 1:
		_, results_IF = tracer_appearance(input_function, meta, return_popt=return_poptmap)
		t_IF, err_IF, gof_IF, popt_ab_IF = results_IF
		_, results_TAC = tracer_appearance(PET, meta, return_popt=return_poptmap)
		t_TAC, err_TAC, gof_TAC, popt_ab_TAC = results_TAC
		if return_poptmap:
			return results_TAC  # TODO: Quick fix.. Needs to be done properly!
		if t_IF > 0 and t_TAC > 0:
			return t_TAC - t_IF
		else:
			return np.nan
	
	# CHECK DATA TYPES
	if not (isinstance(PET, np.ndarray) and PET.ndim == 4): raise TypeError("PET is not a 4D Numpy array")  # PET shape is 4D
	if not (isinstance(mask, np.ndarray) and mask.ndim == 3): raise TypeError("Mask is not a 3D Numpy array")  # Mask shape is 3D
	if not (np.all(np.isin(mask, [0, 1]))): raise TypeError("Mask is not boolean or contains only 0 and 1")  # Mask type is bool or binary
	if not (PET.shape[:-1] == mask.shape): raise TypeError("PET and mask XYZ-shapes do not match")  # PET and mask shapes must match
	if not ((isinstance(input_function, np.ndarray) and input_function.ndim == 1) or
	        (isinstance(input_function, list) and np.ndim(input_function) == 1)): raise TypeError("Input function is not a 1D list or 1D Numpy array")  # Input function shape is 1D
	
	if N_cpu is None: N_cpu = int(multiprocessing.cpu_count() * 2 / 3)  # Number of CPUs to use. Must be less than the total number of CPUs.
	if N_cpu > multiprocessing.cpu_count(): raise ValueError("The number of CPUs used for multiprocessing (" + str(N_cpu) + ") must be less than the total number of CPUs (" + str(multiprocessing.cpu_count()) + ")")
	
	print("Masking...")
	if mask.all() is not None and mask.ndim == 3: coronal, sagittal, axial = np.where(mask)
	else:                                         coronal, sagittal, axial = np.array(np.where(np.ones(PET.shape[0:3])))
	
	PET_shape = PET.shape
	PET_list = np.array([copy.copy(PET[coronal[i], sagittal[i], axial[i], :]) for i in tqdm(range(coronal.size), desc="Listifying")])  # List of masked PET time activity curves
	
	# Input function (IF) appearance time
	_, results_IF = tracer_appearance([input_function], meta)
	t_IF, err_IF, gof_IF, popt_ab_IF = results_IF
	print("Input function onset time: ", t_IF, " +/- ", err_IF, " s.", " (gof: ", gof_IF, ")", sep="")
	
	bunch_size_to_save = 1000
	print("-----\n" + ts.yellow + f"Multiprocessing. Will save results to temporary file every {bunch_size_to_save} voxels calculated" + ts.end)
	if afterburner: print(ts.yellow + "NOTE: afterburner is on. This will rerun the algorithm on failed voxels." + ts.end)
	
	# Loading temporary results
	initialize_temp = False
	if use_temp:
		if os.path.exists(os.path.join(temp_folder, "temp_delaymap.ndjson")):
			print("Loading temp...")
			with open(os.path.join(temp_folder, "temp_delaymap.ndjson"), "r") as fp:
				temp = ndjson.load(fp)
			if len(temp) > 0:
				...  # Do nothing
			else:
				initialize_temp = True
	if not use_temp or not os.path.exists(os.path.join(temp_folder, "temp_delaymap.ndjson")) or initialize_temp:
		temp = []
	
	everything = []
	if use_temp:
		everything += temp
	results_counter_since_last_save = 0
	time1 = time()
	for result in tqdm(dynamic_multiprocessing(PET_list[len(temp):], tracer_appearance, meta, min_processes=3, auto=True), desc="Delaymap", initial=len(temp), total=len(PET_list), disable=False):
		everything += [result[1]]  # Todo: change output of tracer_appearance so as not to have to do this hokus
		
		results_counter_since_last_save += 1
		
		if save_temp and not use_temp and results_counter_since_last_save == 0:  # Reset temp file if not using temp and it's the first bunch
			with open(os.path.join(temp_folder, "temp_delaymap.ndjson"), "w") as fp:
				ndjson.dump([], fp)
		if save_temp and (results_counter_since_last_save >= bunch_size_to_save or len(everything) == len(PET_list[len(temp):])):
			with open(os.path.join(temp_folder, "temp_delaymap.ndjson"), "a") as fp:
				ndjson.dump(everything[-results_counter_since_last_save:], fp)
				fp.write("\n")
			results_counter_since_last_save = 0
	
	duration1 = time() - time1
	print("Total computation time: " + str(timedelta(seconds=duration1).seconds // 3600) + "h" +
	      str(timedelta(hours=-(timedelta(seconds=duration1).seconds // 3600), seconds=duration1).seconds // 60) + "m" +
	      str(timedelta(hours=-(timedelta(seconds=duration1).seconds // 3600), minutes=-(timedelta(hours=-(timedelta(seconds=duration1).seconds // 3600), seconds=duration1).seconds // 60), seconds=duration1).seconds) + "s")
	print("Preparing results...")
	pbar = tqdm(total=6, leave=False)
	pbar.set_description("Prepping")
	
	pbar.set_description("Initializing arrays")
	
	TOI = np.array([everything[i][0] for i in range(len(everything))])  # Note: uncorrected for IDIF delay t_IF. It's done for "delaymap"
	EOI = np.array([everything[i][1] for i in range(len(everything))])
	gof = np.array([everything[i][2] for i in range(len(everything))])
	popt_a = np.array([everything[i][3][0] for i in range(len(everything))])
	popt_b = np.array([everything[i][3][1] for i in range(len(everything))])
	
	pbar.update()
	pbar.set_description("dm")
	
	delaymap = np.zeros(PET.shape[0:3])
	delaymap[coronal, sagittal, axial] = TOI - t_IF  # Subtract IDIF t_IF
	delaymap[coronal[TOI < 0], sagittal[TOI < 0], axial[TOI < 0]] = 0  # Set errors to 0 to avoid seemingly very low delay values
	
	pbar.update()
	pbar.set_description("err")
	
	errormap = np.zeros(PET.shape[0:3])
	errormap[coronal, sagittal, axial] = EOI
	
	pbar.update()
	pbar.set_description("gof")
	
	gofmap = np.zeros(PET.shape[0:3])
	gofmap[coronal, sagittal, axial] = gof
	
	pbar.update()
	pbar.set_description("a")
	
	a_map = np.zeros(PET.shape[0:3])
	a_map[coronal, sagittal, axial] = popt_a
	
	pbar.update()
	pbar.set_description("b")
	
	b_map = np.zeros(PET.shape[0:3])
	b_map[coronal, sagittal, axial] = popt_b
	
	poptmap = [a_map, b_map]
	
	pbar.update()
	pbar.close()
	
	print(tabulate([["Negative onset time", np.sum(TOI == -2)],
	                ["No fits with given conditions", np.sum(TOI == -3)],
	                ["Other fitting errors", np.sum(TOI == -1)],
	                ["Negative gof", np.sum(gof < 0)]],
	               headers=["ERROR OVERVIEW", ""], tablefmt="plain"))
	
	if np.sum(gof < 0) > 10 and afterburner:
		print(f"Redoing errors ({np.sum(gof < 0)}) with 5 nearest neighbor averaging...")
		redo_mask = gofmap < 0
		
		if NN5_low_memory:
			print(ts.yellow + "NN5_low_memory is enabled" + ts.end)
			dm_NN5, err_NN5, gof_NN5, a_map_NN5, b_map_NN5 = [[] for _ in range(5)]
			TAC_NN5 = []
			
			print("Calculating nearest neighbor TACs for bad voxels only...")
			coronal_NN5, sagittal_NN5, axial_NN5 = np.where(redo_mask)
			for coronal_NN5_x, sagittal_NN5_x, axial_NN5_x in zip(coronal_NN5, sagittal_NN5, axial_NN5):
				TAC_NN5.append(np.mean(PET[max(0, coronal_NN5_x - 2):min(PET_shape[0], coronal_NN5_x + 3), max(0, sagittal_NN5_x - 2):min(PET_shape[1], sagittal_NN5_x + 3), max(0, axial_NN5_x - 2):min(PET_shape[2], axial_NN5_x + 3), :], axis=(0, 1, 2)))
			
			print("Running delaymap on bad voxels only...")
			for result_NN5 in tqdm(dynamic_multiprocessing(TAC_NN5, NN5_low_memory_worker, meta, input_function, min_processes=3, auto=True), desc="NN5_L", total=np.sum(gof < 0)):
				t_TAC, err_TAC, gof_TAC, [popt_a_TAC, popt_b_TAC, *_] = result_NN5
				dm_NN5.append(t_TAC)
				err_NN5.append(err_TAC)
				gof_NN5.append(gof_TAC)
				a_map_NN5.append(popt_a_TAC)
				b_map_NN5.append(popt_b_TAC)
			
			coronal_NN5, sagittal_NN5, axial_NN5 = np.where(redo_mask)
			delaymap[coronal_NN5, sagittal_NN5, axial_NN5] = dm_NN5
			errormap[coronal_NN5, sagittal_NN5, axial_NN5] = err_NN5
			gofmap[coronal_NN5, sagittal_NN5, axial_NN5] = gof_NN5
			a_map[coronal_NN5, sagittal_NN5, axial_NN5] = a_map_NN5
			b_map[coronal_NN5, sagittal_NN5, axial_NN5] = b_map_NN5
			poptmap = [a_map, b_map]
		
		else:
			PET_NN5 = nearest_neighbor(PET, "5-space-equal", p_cpu=N_cpu)
			dm_NN5, err_NN5, gof_NN5, [a_map_NN5, b_map_NN5] = create_delaymap(PET_NN5, meta, input_function, redo_mask, N_cpu=N_cpu, return_errormap=True, return_gofmap=True, return_poptmap=True, save_temp=False, use_temp=False, afterburner=False, is_afterburner=True)
			
			coronal_NN5, sagittal_NN5, axial_NN5 = np.where(redo_mask)
			delaymap[coronal_NN5, sagittal_NN5, axial_NN5] = dm_NN5[coronal_NN5, sagittal_NN5, axial_NN5]
			errormap[coronal_NN5, sagittal_NN5, axial_NN5] = err_NN5[coronal_NN5, sagittal_NN5, axial_NN5]
			gofmap[coronal_NN5, sagittal_NN5, axial_NN5] = gof_NN5[coronal_NN5, sagittal_NN5, axial_NN5]
			a_map[coronal_NN5, sagittal_NN5, axial_NN5] = a_map_NN5[coronal_NN5, sagittal_NN5, axial_NN5]
			b_map[coronal_NN5, sagittal_NN5, axial_NN5] = b_map_NN5[coronal_NN5, sagittal_NN5, axial_NN5]
			poptmap = [a_map, b_map]
	
	returning = []
	returning.append(delaymap)
	if return_errormap: returning.append(errormap)
	if return_gofmap:   returning.append(gofmap)
	if return_poptmap:  returning.append(poptmap)
	if not (return_errormap and return_gofmap and return_poptmap): returning = returning[0]
	if not is_afterburner: print("DELAYMAP DONE!")
	return returning


if __name__ == "__main__":
	print("Import this file in Python and run create_delaymap(PET: np.ndarray, meta: dict, input_finction: np.array, mask: np.ndarray, N_cpu: int)")
	# TODO: Add support for .npy and .npz
	parser = argparse.ArgumentParser(prog="Delaymap", description="Create delaymap from dynamic PET and an input function (IF). Produces a map of estimated delay times between each voxel and the input function in seconds: delay = time_voxel - time_IF. Use a mask to limit calculation to either the edge of the patient or an organ.")
	# User can choose path to a custom file (either DICOM folder or Nifti file)
	parser.add_argument("PETpath", type=str, required=True, help="Path to dynamic PET file (either DICOM folder or Nifti file). Supports Nifti (.nii, .nii.gz) files" + ", Numpy (.npy, .npz) files" * 0 + " and DICOM (.dcm) folders.")
	# Path to input function.
	parser.add_argument("-i", "--ifpath", type=str, required=True, help="Path to input function.")
	# Mask path
	mask_group = parser.add_mutually_exclusive_group(required=False)
	mask_group.add_argument("-a", "--maskpath", type=str, default=None, help="Optional path to custom mask.")
	parser.add_argument("-e", "--maskedge", action="store_true", help="Find and use the edge of the mask/region to mask the PET scans. Different from the raw mask, this will not have unmasked areas inside the patient. Must be given together with --maskname.")
	mask_group.add_argument("-b", "--maskpetthreshold", type=float, nargs="?", const=10e3, help="If no mask is provided, a threshold can be set for the max TAC value, which will be used to mask the PET scan. Default if flag is given without a value: 10e3.")
	# Custom output directory
	parser.add_argument("-o", "--output", type=str, nargs="?", default=None, const="./", help="Output directory for result and temp files. Defaults to './' if flag is given without a path, and '/raid/derivatives/.../delaymap/' if flag is not given at all. Use this if no write permission to /raid/derivatives/.")
	# Custom filename
	parser.add_argument("-f", "--filename", type=str, default=None, help="Provide a custom filename to the delaymap. Default is 'delaymap_[--maskname]_[--maskregion]_[#].nii'.")
	# Number of CPUs to use
	parser.add_argument("-n", "--ncpus", type=int, default=multiprocessing.cpu_count(), help=f"Specify number of CPUs to use in multiprocessing. Default varies around 80%% based on the time of day, with more during the night and weekend. Currently: {N_CPUs_to_use()}.")
	# Use temporary delaymap file
	parser.add_argument("-t", "--temp", action="store_true", help="In case the script stopped, use the temporary delaymap result file ('temp_delaymap.ndjson' located in the output dir) to restart the calculations from where it stopped.")
	# Disable afterburner
	parser.add_argument("--disable_afterburner", action="store_false", help="Disable afterburner. The afterburner runs the delaymap algorithm again on failed voxels by averaging their nearest neighbors. Takes an additional 5-10 mins.")
	# Low memory for NN5 calculation (won't make a copy of PET file in memory with all voxels calculated, but only passes NN5 TACs to create_delaymap)
	# parser.add_argument("-l", "--NN5_low_memory", action="store_true", help="Use low memory for nearest neighbor averaging.")
	# Save only delaymap or all maps (dm, err, gof, a, b)
	parser.add_argument("-s", "--save_all_maps", action="store_true", help="Save all maps: delaymap, errormap, goodness-of-fit-map, parameter maps a and b.")
	# Keep temporary files instead of deleting them
	parser.add_argument("-k", "--keep_temp", action="store_true", help="Keep all temporary files and folders in the output directory.")
	# Debug mode (a few extra print statements)
	parser.add_argument("-d", "--debug", action="store_true", help="Debug mode. Prints extra information.")
	
	args = parser.parse_args()
	if args.debug:
		for arg, value in vars(args).items():
			print(ts.magenta + f"{arg}:\t{value}" + ts.end)
	
	# Get arguments
	PETPATH = args.PETpath
	OUTPUTDIR = args.output
	OUTPUTFILENAME = args.filename
	IFPATH = args.ifpath
	MASKEDGE = args.maskedge
	MASKPATH = args.maskpath
	MASKPETTHRESHOLD = args.maskpetthreshold
	if MASKPETTHRESHOLD: MASKNAME = "MASKPET" + str(int(MASKPETTHRESHOLD))
	NCPUS = args.ncpus
	USETEMPDM = args.temp
	AFTERBURNER = args.disable_afterburner
	SAVEALLMAPS = args.save_all_maps
	KEEPTEMP = args.keep_temp
	DEBUG = args.debug
	
	
	if NCPUS >= multiprocessing.cpu_count() - 1:
		raise ValueError("The selected number of CPUs can't be greater than the available number of CPUs (" + multiprocessing.cpu_count() + ").")
	
	
	PET_filetype = None
	MASK_filetype = None
	MASK_is_in_PETspace = None
	# MASK_PETspace_dicom_dir_is_temp = False
	append_datetime_for_temps = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S')
	load_PET_using_nifti = os.path.splitext(PETPATH)[0] == ".nii"
	load_PET_using_dcm = not os.path.splitext(PETPATH)[0] == ".nii"
	
	# PET path
	if PETPATH:
		if not os.path.exists(PETPATH):
			raise FileNotFoundError("PET path does not exist")
		
		if PETPATH.endswith((".nii", ".nii.gz")):
			PET_filetype = "nifti"
		elif len(glob.glob(os.path.join(PETPATH, "*.dcm"))) > 0:
			PET_filetype = "dicom"
	
	
	# Input function path
	if IFPATH:
		if not os.path.exists(IFPATH):
			raise FileNotFoundError("The provided input function path does not exist.")
	
	elif not IFPATH:
		raise FileNotFoundError("Desired input function names could not be found. Provide the path manually using -i.")
	
	# Mask path
	if MASKPATH:
		if not os.path.exists(MASKPATH):
			raise FileNotFoundError("Mask path does not exist")
		
		if MASKPATH.endswith((".nii", ".nii.gz")):
			MASK_filetype = "nifti"
		elif len(glob.glob(os.path.join(MASKPATH, "*.dcm"))) > 0:
			MASK_filetype = "dicom"
	
	elif not MASKPATH and MASKNAME == "MASKPET":
		...  # Mask is created from PET (taken care of when loading). No other mask needed.
	
	
	# Output directory
	if OUTPUTDIR:
		if not os.path.isdir(OUTPUTDIR):
			os.mkdir(OUTPUTDIR)
	
	elif not OUTPUTDIR:
		if PETPATH:
			OUTPUTDIR = os.path.dirname(PETPATH)
		else:
			raise FileNotFoundError("Output directory couldn't be found. Please provide an output directory using -o.")
	
	
	# Output path and filename
	def output_filepath(output_path, append=""):
		extensions = [".nii", ".nii.gz", ".npy", ".npz"]
		if output_path.endswith(tuple(extensions)):  # If it's a file with one of the supported extensions
			for extension in extensions:
				if output_path.endswith(extension):
					directory, filename = os.path.split(output_path)[0], os.path.split(output_path)[1].split(extension)[0]
					break
		elif os.path.isdir(output_path):  # If it's an existing folder
			directory = output_path
			filename = "delaymap"
			if MASKNAME:
				filename += f"_{MASKNAME}"
			extension = ".nii"
		else:
			raise ValueError("Output path must be a one of the supported file types (" + ", ".join(extensions) + "), or an existing directory")
		
		new_output_path = os.path.join(directory, filename + str(append) + extension)  # If this is changed, remember to update the one below as well!
		suffix_counter = 0
		while os.path.exists(new_output_path):
			new_output_path = os.path.join(directory, filename + str(append) + f"_{suffix_counter}" + extension)  # If this is changed, remember to update the one above as well!
			suffix_counter += 1
		return new_output_path
	
	
	if not OUTPUTFILENAME:
		OUTPUTPATH = output_filepath(OUTPUTDIR)
		OUTPUTFILENAME = os.path.basename(OUTPUTPATH)
	elif OUTPUTFILENAME:
		if not OUTPUTFILENAME.endswith(".nii"):
			OUTPUTFILENAME += ".nii"
		OUTPUTPATH = output_filepath(os.path.join(OUTPUTDIR, OUTPUTFILENAME))
		OUTPUTFILENAME = os.path.basename(OUTPUTPATH)
	
	if not OUTPUTPATH:  # Just in case nothing was found...
		OUTPUTPATH = output_filepath("./")
		OUTPUTFILENAME = os.path.basename(OUTPUTPATH)
		print(f"Outputpath defaults to {OUTPUTPATH}")
	
	# Print all paths
	print("Paths found:")
	print(tabulate([["PET", PETPATH],
	                ["IF", IFPATH],
	                ["Mask", MASKPATH],
	                ["Output", OUTPUTPATH]], tablefmt="simple"))
	
	# Check if paths exist
	for path in [PETPATH, IFPATH, MASKPATH]:
		if path and not os.path.exists(path):
			raise FileNotFoundError(path + " not found")
	
	# Create temporary Nifti file from DICOM folder in OUTPUT folder
	if MASK_filetype == "dicom":
		if not MASK_is_in_PETspace:
			if PET_filetype != "dicom":
				user_ans = input(ts.yellow + "WARNING: Mask must be converted to PET space using PET dicoms. The provided PET path doesn't point to a DICOM dir. Do still you want to continue?" + ts.end + " [y/any key]: ")
				if user_ans != "y":
					raise TypeError("PET path must be a DICOM dir in order to convert mask to PET space")
				elif user_ans == "y":
					MASK_is_in_PETspace = True
			
			else:
				print("Converting mask to PET space...")
				MASK_PETspace_dicom_dir = os.path.join(OUTPUTDIR, "temp_MASK_PETspace_dcm")
				try: os.mkdir(MASK_PETspace_dicom_dir)
				except FileExistsError:
					user_ans = None
					tries_left = 2
					while user_ans not in ["y", "n"] and tries_left > 0:
						if tries_left < 2:  print(ts.yellow + "Didn't get that. Try one last time." + ts.end)
						user_ans = input(ts.yellow + "Trying to create a temp dir for mask dicoms in PET space, but 'temp_MASK_PETspace_dcm' already exists. Do you want to delete it and create a new?" + ts.end + " [y/n] ([n] exits): ")
						tries_left -= 1
					if user_ans == "y":
						print("Deleting temp mask directory and creating a new...")
						shutil.rmtree(MASK_PETspace_dicom_dir)
						os.mkdir(MASK_PETspace_dicom_dir)
					elif user_ans == "n" or tries_left <= 0:
						print("Exiting")
						exit()
				
				# MASK_PETspace_dicom_dir_is_temp = True
				print(ts.blue);
				os.system("python3 /raid/scripts/quadra_reslice.py" + " " + MASKPATH + " " + PETPATH + " " + MASK_PETspace_dicom_dir);
				print(ts.end)
				
				if len(glob.glob(os.path.join(MASK_PETspace_dicom_dir, "*dcm"))) == 0:
					raise FileNotFoundError("Something went wrong while converting the mask to PET space.")
		# MASK_is_in_PETspace = True  # Should still be False!
		
		if MASK_is_in_PETspace:
			MASK_PETspace_dicom_dir = MASKPATH
		
		print("Creating MASK Nifti file from DICOM folder...")
		temp_MASKnii_for_delaymap_filename = "temp_MASKnii_for_delaymap_" + append_datetime_for_temps
		print(ts.blue);
		os.system("dcm2niix -f " + temp_MASKnii_for_delaymap_filename + " -o " + OUTPUTDIR + " " + MASK_PETspace_dicom_dir);
		print(ts.end)
		MASKPATH = os.path.join(OUTPUTDIR, temp_MASKnii_for_delaymap_filename + ".nii")
		print("Temp mask path:", MASKPATH)
	
	if PET_filetype == "dicom" and load_PET_using_nifti:
		temp_PETnii_for_delaymap_filename = "temp_PETnii_for_delaymap_" + append_datetime_for_temps
		print("Creating PET Nifti file from DICOM folder...")
		print(ts.blue);
		os.system("dcm2niix -f " + temp_PETnii_for_delaymap_filename + " -o " + OUTPUTDIR + " " + PETPATH);
		print(ts.end)
		PETPATH = os.path.join(OUTPUTDIR, temp_PETnii_for_delaymap_filename + ".nii")
		print("Temp PET path:", PETPATH)
	
	if PET_filetype == "dicom" and load_PET_using_dcm:
		temp_PETnii_for_delaymap_filename = "temp_PETnii_for_delaymap_" + append_datetime_for_temps
		print("Creating PET metadata .json file...")
		print(ts.blue);
		os.system("dcm2niix -b o -f " + temp_PETnii_for_delaymap_filename + " -o " + OUTPUTDIR + " " + PETPATH);
		print(ts.end)
		PETMETAPATH = os.path.join(OUTPUTDIR, temp_PETnii_for_delaymap_filename + ".json")
		print("Temp PET meta path:", PETMETAPATH)
	
	# Just a sanity check
	if load_PET_using_nifti and not (PETPATH.endswith((".nii", ".nii.gz")) or (MASKPATH is not None and MASKPATH.endswith((".nii", ".nii.gz")))):
		raise FileNotFoundError("PETPATH or MASKPATH is not Nifti. Something might have gone wrong while converting from dicom to nifti.")
	
	# Load data
	print("Loading PET (" + PETPATH + ")...")
	if load_PET_using_nifti:
		PET, PET_affine = load_data(PETPATH, silent=True)
	elif load_PET_using_dcm and SERIESUID:
		PET, PET_planes = load_PET_dcm(SERIESUID)
	
	if False and SERIESUID and getdcmmeta(SERIESUID).tracer.lower() == "dotatate":
		print("Tracer is DOTATATE. Using nearest neighbor.")
		PET = nearest_neighbor(PET, "3-space-equal")  # TODO: this uses a lot of memory (~80%). Maybe use psutil.virtual_memory() to determine?
	
	if load_PET_using_nifti:
		PETMETAPATH = re.split(r'(\.nii.*)', PETPATH, maxsplit=1)[0] + ".json"
		tries_left = 2
		while not os.path.exists(PETMETAPATH) and tries_left > 0:
			PETMETAPATH = input(f"PET meta file ({os.path.basename(PETMETAPATH)[0]}) could not be found in the directory. You can provide the path here ({tries_left} tries left): ")
			tries_left -= 1
	meta = load_data(PETMETAPATH, silent=True)
	
	print("Loading input function (" + IFPATH + ")...")
	IF = load_data(IFPATH, silent=True)
	
	if MASKNAME == "MASKPET" and MASKPETTHRESHOLD:
		MASK = np.max(PET, axis=-1) > MASKPETTHRESHOLD
	elif MASKPATH:
		print("Loading mask (" + MASKPATH + ")...")
		MASKMETAPATH = re.split(r'(\.nii.*)', MASKPATH, maxsplit=1)[0] + ".json"
		[MASK, MASK_affine] = load_data(MASKPATH, silent=True)
		if load_PET_using_dcm:
			PET_affine = MASK_affine
		MASK = MASK != 0
	else:
		MASK = None
	
	if MASKEDGE and MASK is not None:
		print("Finding the edge of the mask...")
		MASK = edge(MASK)
	
	if DEBUG:
		print("PET shape:       ", np.shape(PET))
		print("MASK shape & sum:", np.shape(MASK), np.sum(MASK))
	
	# Delete temporary files
	if PET_filetype == "dicom" and not KEEPTEMP:
		for file in [PETPATH, PETMETAPATH, re.split(r'(\.nii.*)', PETPATH, maxsplit=1)[0] + "_ROI1.nii"]:
			if os.path.exists(file) and OUTPUTDIR in file:
				print("Deleting temp PET Nifti file:", file)
				try:
					os.remove(file)
				except Exception as e:
					print("os.remove() error:", e)
	if MASK_filetype == "dicom" and not KEEPTEMP:
		for file in [MASKPATH, MASKMETAPATH, re.split(r'(\.nii.*)', MASKPATH, maxsplit=1)[0] + "_ROI1.nii"]:
			if os.path.exists(file) and OUTPUTDIR in file:
				print("Deleting temp mask Nifti file:", file)
				try:
					os.remove(file)
				except Exception as e:
					print("os.remove() error:", e)
		if MASK_PETspace_dicom_dir and OUTPUTDIR in MASK_PETspace_dicom_dir:
			print("Deleting temp directory with mask dicom in PET space:", MASK_PETspace_dicom_dir)
			# input(ts.yellow + "Press [Enter] to accept" + ts.end +" (" + MASK_PETspace_dicom_dir +")")
			shutil.rmtree(MASK_PETspace_dicom_dir)
	
	if np.sum(MASK) < 1: raise IndexError("Mask contains only zeros.")
	
	# Create delaymap
	print(ts.green + "Creating delaymap..." + ts.end)
	dm, err, gof, [a_map, b_map] = create_delaymap(PET, meta, IF, mask=MASK, N_cpu=NCPUS, NN5_low_memory=True, return_errormap=True, return_gofmap=True, return_poptmap=True, save_temp=True, use_temp=USETEMPDM, temp_folder=os.path.dirname(OUTPUTPATH), afterburner=AFTERBURNER)
	
	print(ts.green + "Saving map" + "s" * SAVEALLMAPS + "..." + ts.end)
	maps = [dm, err, gof, a_map, b_map] if SAVEALLMAPS else [dm]
	map_names = ["", "-err", "-gof", "-a", "-b"] if SAVEALLMAPS else [""]
	for map_, map_name in zip(maps, map_names):
		map_path = output_filepath(OUTPUTPATH, append=map_name)
		print("map path:", map_path)
		if map_path.endswith((".nii", ".nii.gz")):
			savenii(map_, affine=PET_affine, filepath=map_path)
		elif map_path.endswith(".npy"):
			np.save(map_path, map_)
		elif map_path.endswith(".npz"):
			np.savez_compressed(map_path, map_)
	
	if os.path.exists(os.path.join(OUTPUTDIR, "temp_delaymap.ndjson")) and not KEEPTEMP:
		print("Deleting temp result file...")
		os.remove(os.path.join(OUTPUTDIR, "temp_delaymap.ndjson"))
	
	print(ts.green + "Delaymap done!" + ts.end)

