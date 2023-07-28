from __future__ import print_function

# Author: Christopher Lin <clin5@bidmc.harvard.edu>

from collections import OrderedDict
from scipy.stats import ttest_1samp
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing as mp
from numba import jit
import time
import os
from tqdm import tqdm
from nimlab import functions as nimfs
from queue import Queue


class ConnectomeSubject:
    """Object that represents a connectome subject and associated ROIs

    Since loading a connectome file into memory incurs a significant time penalty, we use the connectome subject as the unit of computation rather than the ROI.

    Attributes:
        roi_connectome_file (str): File path to roi connectome subject file. We pass paths since we do not want to have to load connectome files until they are needed.
        roi_norms (str): File path to roi connectome subject norm file.
        brain_connectome_file (str): File path to brain connectome subject file. We pass paths since we do not want to have to load connectome files until they are needed.
        brain_norms (str): File path to brain connectome subject norm file.
        rois (dict of str : ndarray): Dictionary of ROIs. The string specifies the name of the ROI (often the filename), and the ndarray is the masked and flattened ROI image.
        roi_connectome_type (str): Space of ROIs ("volume" or "surface")
        brain_connectome_type (str): Space of Brain ("volume" or "surface")
        same_connectome (bool): Boolean to determine whether the ROI and Brain connectomes are the same to potentially prevent double-loading of connectome files
        warning_flag (bool): Boolean to determine whether to display warning messages (True will display warnings).
    """

    def __init__(
        self,
        roi_connectome_file,
        brain_connectome_file,
        rois,
        roi_connectome_type,
        brain_connectome_type,
        same_connectome,
        warning_flag,
    ):
        self.roi_connectome_file = roi_connectome_file[0]
        self.roi_norms = roi_connectome_file[1]
        self.brain_connectome_file = brain_connectome_file[0]
        self.brain_norms = brain_connectome_file[1]
        self.rois = rois
        self.roi_type = roi_connectome_type
        self.brain_type = brain_connectome_type
        self.same_type = same_connectome
        self.warning_flag = warning_flag


def make_fz_maps(subject):
    """Calculates the Fischer transformed average Pearson R correlation between a roi region and the rest of the voxels in a
    connectome file.

    Args:
        subject (ConnectomeSubject): The connectome subject file (with associated ROIs) to compute fz maps for.

    Returns:
       (dict of str : ndarray): Dictionary mapping the ROI name to its fz map
    """
    roi_connectome_mat = np.load(subject.roi_connectome_file).astype(np.float32)
    roi_connectome_norms_mat = np.load(subject.roi_norms).astype(np.float32)

    if not subject.same_type:
        brain_connectome_mat = np.load(subject.brain_connectome_file).astype(np.float32)
        brain_connectome_norms_mat = np.load(subject.brain_norms).astype(np.float32)

    fz_maps = {}
    for roi_key in subject.rois.keys():
        roi = subject.rois[roi_key]
        if roi_connectome_mat.shape[1] != roi.shape[1]:
            raise ValueError(
                "ROI not masked with same mask as connectome file. \
                    Connectome file has "
                + str(roi_connectome_mat.shape[1])
                + " voxels, while \
                    the ROI has "
                + str(roi.shape[1])
            )

        # Mask connectome time courses, find mean tc of masked area
        roi_mean_tc = extract_avg_signal(roi_connectome_mat, roi)

        # Catch problems where roi connectome subjects and brain connectome subjects don't have the same length timecourse signal
        try:
            if subject.same_type:
                corr_num = np.dot(roi_connectome_mat.T, roi_mean_tc)
                corr_denom = roi_connectome_norms_mat * np.linalg.norm(roi_mean_tc)
            else:
                corr_num = np.dot(brain_connectome_mat.T, roi_mean_tc)
                corr_denom = brain_connectome_norms_mat * np.linalg.norm(roi_mean_tc)
        except ValueError:
            if subject.same_type:
                min_TC = np.min([roi_connectome_mat.shape[0], roi_mean_tc.shape[0]])
            else:
                min_TC = np.min([brain_connectome_mat.shape[0], roi_mean_tc.shape[0]])
            print(
                f"WARNING: ROI Connectome subject {os.path.basename(subject.roi_connectome_file)} and Brain Connectome subject\
            {os.path.basename(subject.brain_connectome_file)} do not have the same length Timecourse Signal. Instead using the\
            smallest common timecourse signal length ({min_TC}) to compute correlations."
            )

            if subject.same_type:
                corr_num = np.dot(
                    roi_connectome_mat.T[:, :min_TC],
                    roi_mean_tc[
                        :min_TC,
                    ],
                )
                corr_denom = np.linalg.norm(
                    roi_connectome_mat[:min_TC, :], axis=0
                ) * np.linalg.norm(
                    roi_mean_tc[
                        :min_TC,
                    ]
                )
            else:
                corr_num = np.dot(
                    brain_connectome_mat.T[:, :min_TC],
                    roi_mean_tc[
                        :min_TC,
                    ],
                )
                corr_denom = np.linalg.norm(
                    brain_connectome_mat[:min_TC, :], axis=0
                ) * np.linalg.norm(
                    roi_mean_tc[
                        :min_TC,
                    ]
                )

        np.seterr(invalid="ignore")
        corr = corr_num / corr_denom
        if subject.warning_flag:
            if corr.max() > 1:
                print(roi_key)
                print("Unexpected corr value: " + str(corr.max()))

        # Make sure fisher z transform is taking in valid values
        corr[np.isnan(corr)] = 0
        fz = np.arctanh(corr)

        # Fix infinite values in the case of single voxel autocorrelations
        finite_max = fz[np.isfinite(fz)].max()
        fz[np.isinf(fz)] = finite_max
        fz_maps.update([(roi_key, fz)])

    return fz_maps


def extract_avg_signal(connectome_mat, roi_mat):
    """Extracts a single time course from a region in a connectome file specified by an ROI

    The current extraction method is to mask out all voxels where the ROI image is 0, multiply the remaining connectome
    voxs by the weights specified by the ROI image, and then average all voxel timecourses together.
    NOTE: The ROI image can be weighted.

    Args:
        connecotme_mat(ndarray): Masked and flattened connectome subject image
        roi_mat(ndarray): Masked and flattened ROI image of the same length as the connectome subject

    Returns:
        ndarray: Extracted signal. Has same shape as the connectome file and roi image
    """
    roi_masked_tc = connectome_mat[:, roi_mat[0, :] != 0]
    roi_masked_tc = roi_masked_tc * roi_mat[roi_mat[0:] != 0]
    roi_mean_tc = np.nanmean(roi_masked_tc, axis=1)

    return roi_mean_tc


def make_fz_maps_to_queue(
    subject,
    result_queue,
    output_folder,
    ind_file_output="",
    single_connectome_subject=False,
):
    """Calculate FZ maps for a single ConnectomeSubject and push the results to a queue

    Args:
        subject (ConnectomeSubject): ConnectomeSubject to process
        result_queue (Queue): Queue to push result maps to. The queue is usually monitored by a map maker process.
        output_folder (str): File path to output maps.
        ind_file_output (str, optional): File path to output individual FZ maps. Defaults to '', which
            produces no output.
        single_connectome_subject (bool): If True, output Fz files only. Defaults to False.
    """
    connectome_fname = subject.roi_connectome_file.split("/")[-1]
    fz_maps = make_fz_maps(subject)
    for roi_key in fz_maps.keys():
        result_queue.put((roi_key, fz_maps[roi_key]))
        if ind_file_output:
            # Probably making some unsafe assumptions about ROI naming, but
            # I don't see any better way to do it.
            if subject.brain_type is "surface":
                roi_name = list(
                    nimfs.lcs(
                        roi_key.split("|")[0].split("/")[-1].split(".")[0],
                        roi_key.split("|")[1].split("/")[-1].split(".")[0],
                    )
                )[0].split("_", 1)[1]
            else:
                roi_name = roi_key.split("/")[-1].split(".")[0]
            roi_ind_directory = ind_file_output + "/" + roi_name
            if not os.path.exists(roi_ind_directory):
                os.makedirs(roi_ind_directory)
            out_fname = roi_ind_directory + "/" + connectome_fname
            np.save(out_fname, fz_maps[roi_key])
        if single_connectome_subject:
            # Probably making some unsafe assumptions about ROI naming, but
            # I don't see any better way to do it.
            if subject.brain_type is "surface":
                roi_name = list(
                    nimfs.lcs(
                        roi_key.split("|")[0].split("/")[-1].split(".")[0],
                        roi_key.split("|")[1].split("/")[-1].split(".")[0],
                    )
                )[0].split("_", 1)[1]
            else:
                roi_name = roi_key.split("/")[-1].split(".")[0]
            roi_ind_directory = output_folder + "/" + roi_name
            if not os.path.exists(roi_ind_directory):
                os.makedirs(roi_ind_directory)
            out_fname = roi_ind_directory + "/" + connectome_fname
            np.save(out_fname, fz_maps[roi_key])


def gen_welford_maps_from_queue(n_connectome, n_roi, result_queue, welford_maps_queue):
    """Creates welford maps from a queue filled with fz maps from make_fz_maps()

    Args:
        n_connectome(int): Number of connectome files
        n_roi(int): Number of ROIs
        result_queue(Queue): Queue of fz maps
        welford_maps_queue(Queue): Queue onto which welford maps generated are pushed
    """
    current_count = 0
    welford_maps = {}
    total_maps = n_connectome * n_roi
    with tqdm(total=total_maps) as pbar:
        while current_count < total_maps:
            roi_result = result_queue.get()
            roi_key = roi_result[0]
            roi_map = roi_result[1]
            if roi_key not in welford_maps.keys():
                fz_welford_result = welford_update_map(
                    np.zeros(roi_map.shape[0], dtype=(np.float32, 3)), roi_map
                )
            else:
                fz_welford_result = welford_update_map(welford_maps[roi_key], roi_map)
            welford_maps.update([(roi_key, fz_welford_result)])
            pbar.update(1)
            current_count += 1
    print("Welford maps constructed")
    for m in welford_maps.items():
        welford_maps_queue.put(m)


def calculate_maps(
    subjects,
    num_workers,
    output_folder,
    ind_file_output="",
    single_connectome_subject=False,
):
    """Calculates AvgR_Fz, AvgR, and T maps for a set of subjects

    Args:
        subjects (ConnectomeSubject[]): List of connectome subjects with associated ROIs
        num_workers (int): Number of workers processes
        output_folder (str): File path to output maps to.
        ind_file_output (str, optional): File path to output individual FZ maps.
            Defaults to '', which produces no output.
        single_connectome_subject (bool): If True, only single connectome subject, output Fz map only.
    Returns:
        (dict, dict, dict): Dictionaries of AvgR_Fz maps, AvgR maps, T maps
        or
        None: If single connectome subject
    """
    # TODO: refactor tuples scheme, since we are no longer using pool/map
    os.nice(19)
    result_queue = mp.Queue()
    final_maps_queue = mp.Queue()

    if len(subjects) > 1:
        map_maker = mp.Process(
            target=gen_welford_maps_from_queue,
            args=(len(subjects), len(subjects[0].rois), result_queue, final_maps_queue),
        )
        map_maker.start()
        num_pool_workers = num_workers
        print("Using pool of " + str(num_pool_workers) + " workers")
        pool = []
        for s in subjects:
            while len(pool) >= num_pool_workers:
                for w in pool:
                    if w.is_alive() is False:
                        w.join()
                        pool.remove(w)
                time.sleep(1)
            job = mp.Process(
                target=make_fz_maps_to_queue,
                args=(
                    s,
                    result_queue,
                    output_folder,
                    ind_file_output,
                    single_connectome_subject,
                ),
            )
            pool.append(job)
            job.start()
        fz_welford_maps = {}
        for _ in range(0, len(subjects[0].rois)):
            welford_map = final_maps_queue.get()
            fz_welford_maps.update([welford_map])
        map_maker.join()

        avgR_fz_maps, avgR_maps, T_maps = make_stat_maps(fz_welford_maps, len(subjects))
        return avgR_fz_maps, avgR_maps, T_maps
    else:
        s = subjects[0]
        result_queue = Queue()
        make_fz_maps_to_queue(
            s, result_queue, output_folder, ind_file_output, single_connectome_subject
        )
        return None


def make_stat_maps(fz_welford_maps, num_subjects):
    """Generates statistical maps from welford maps

    Args:
        fz_welford_maps(dict of str : ndarray): Dictionary mapping ROI names to welford maps.
        num_subjects (int): Number of subjects.

    Returns:
        (dict, dict, dict): avgR_fz maps, avgR_maps, T_maps
    """
    avgR_fz_maps = {}
    avgR_maps = {}
    T_maps = {}
    print("Constructing statistical maps")
    for roi_key in tqdm(fz_welford_maps.keys()):
        fz_welford_final = np.asarray(welford_finalize_map(fz_welford_maps[roi_key]))

        # construct avgR_fz map
        avgR_fz = fz_welford_final[:, 0]
        avgR_fz_maps.update([(roi_key, avgR_fz)])

        # construct avgR map
        avgR = np.tanh(avgR_fz)
        avgR_maps.update([(roi_key, avgR)])

        # construct T map
        variance = fz_welford_final[:, 2]
        num_samples = num_subjects
        t_test_denom = np.sqrt(variance / num_samples)
        T = avgR_fz / t_test_denom
        T_maps.update([(roi_key, T)])

    return avgR_fz_maps, avgR_maps, T_maps


def calculate_roi_matrix(subjects, num_workers):
    """Calculates and ROI-ROI matrix correlation for a set of connectome subjects (with associated ROIs)

    Args:
        subjects(ConenctomeSubject[]): set of subjects with ROIs
        num_workers(int): Number of worker processes

    Returns:
        ndarray, ndarray, ndarray, roi_names: AvgR_Fz matrix, AvgR matrix, T matrix, list of ROI names in order
    """
    os.nice(19)
    pool = Pool(num_workers)
    fz_list = list(tqdm(pool.imap(make_fz_matrix, subjects), total=len(subjects)))
    fz_array = np.asarray(fz_list)
    avgR_fz = np.mean(fz_array, axis=0)
    avgR = np.tanh(avgR_fz)
    T = ttest_1samp(fz_array, 0, axis=0)[0]

    roi_names = subjects[0].rois.keys()

    return avgR_fz, avgR, T, roi_names


def make_fz_matrix(subject):
    """Creates a fz transformed matrix of ROI-ROI correlations for a single subject

    Args:
        subject(ConnectomeSubject): Subject with associated ROIs

    Returns:
        ndarray: R_Fz matrix
    """
    roi_tcs = OrderedDict()
    connectome_mat = np.load(subject.roi_connectome_file)
    for roi_key in subject.rois.keys():
        roi_tcs.update(
            [(roi_key, extract_avg_signal(connectome_mat, subject.rois[roi_key]))]
        )
    corrs = np.corrcoef(list(roi_tcs.values()))
    corrs[np.isnan(corrs)] = 0
    np.seterr(divide="ignore")
    fz = np.arctanh(corrs)

    return fz


@jit(nopython=True)
def welford_update_map(existingAggregateMap, newMap):
    """Update a Welford map with data from a new map

    Args:
        existingAggregateMap (ndarray): An existing Welford map, which is an array of
            3-element tuples.
        newMap (ndarray): New data to incorporate into the existing Welford map

    Returns:
        ndarray: Updated Welford map
    """
    newAggregates = []
    for i in range(0, existingAggregateMap.shape[0]):
        newAggregate = welford_update(existingAggregateMap[i], newMap[i])
        newAggregates.append(newAggregate)

    return np.asarray(newAggregates)


# The following two methods are from wikipedia:
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
@jit(nopython=True)
def welford_update(existingAggregate, newValue):
    """Update a Welford tuple with a new value. Welford tuples are 3-element tuples, which contain the
    arguments of Welford's update algorithm. See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Args:
        existingAggregate ((float, float, float)): Welford's tuple to be upgraded
        newValue (float): New data to update tuple with

    Returns:
        (float, float, float): Updated Welford's tuple
    """
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)


@jit(nopython=True)
def welford_finalize_map(existingAggregateMap):
    """Convert a Welford map into maps of arrays containing the statistics
    [mean, variance, sampleVariance].

    Args:
        existingAggregateMap (ndarray): Map of Welford tuples

    Returns:
        ndarray: Map of statistics
    """
    finalMap = []
    for i in range(0, existingAggregateMap.shape[0]):
        finalized_map = welford_finalize(existingAggregateMap[i])
        finalMap.append(finalized_map)
    return finalMap


# retrieve the mean, variance and sample variance from an aggregate
@jit(nopython=True)
def welford_finalize(existingAggregate):
    """Convert a single Welford tuple into an array of statistics

    Args:
        existingAggregate ((float, float, float)): Welford tuple

    Returns:
        ndarray: Array with elements [mean, variance, sampleVariance]
    """
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    return np.asarray([mean, variance, sampleVariance], dtype=np.float32)
