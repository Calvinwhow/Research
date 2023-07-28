from nimlab import connectomics as cs
from nilearn import maskers
import numpy as np
import Queue
from glob import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Collate hpc output files")
    parser.add_argument("-i", metavar = 'input', help="Input folder", required = True)
    parser.add_argument("-o", metavar = 'output', help="Output folder", required = True)
    parser.add_argument("-min", metavar = 'input_mask', help="Input mask", required = True)
    args = parser.parse_args()

    files = glob(args.i + "/*")

    fz_queue = Queue.Queue()

    test_subject = np.load(files[0], allow_pickle = True).item()
    
    for sub in files:
        subject = np.load(sub, allow_pickle=True).item()
        fz_queue.put(subject)

    welford_queue = Queue.Queue()

    n_roi = len(test_subject.keys())
    n_vox = test_subject[test_subject.keys()[0]].shape[0]

    cs.gen_welford_maps_from_queue(len(files), n_roi, n_vox, fz_queue, welford_queue)

    fz_welford_maps = {}
    for i in range(0, n_roi):
        welford_map = welford_queue.get()
        fz_welford_maps.update([welford_map])
    avgR_fz_maps, avgR_maps, T_maps = cs.make_stat_maps(fz_welford_maps)

    output_folder = args.o
    brain_masker = maskers.NiftiMasker(args.min, standardize = False)
    brain_masker.fit()

    for key in avgR_fz_maps.keys():
        fname = key.split('/')[-1].split('.')[0] + '_AvgR_Fz.nii.gz'
        brain_masker.inverse_transform(avgR_fz_maps[key]).to_filename(output_folder + '/' + fname)
    for key in avgR_maps.keys():
        fname = key.split('/')[-1].split('.')[0] + '_AvgR.nii.gz'
        brain_masker.inverse_transform(avgR_maps[key]).to_filename(output_folder + '/' + fname)
    for key in T_maps.keys():
        fname = key.split('/')[-1].split('.')[0] + '_T.nii.gz'
        brain_masker.inverse_transform(T_maps[key]).to_filename(output_folder + '/' + fname) 



