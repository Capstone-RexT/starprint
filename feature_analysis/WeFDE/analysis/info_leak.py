# -*- coding: utf-8 -*-
"""
Main project file which performs info-leak measure

- fingerprint_modeler.py: individual leakage analysis
- mi_analyzer.py: combined leakage analysis.
- data_utils.py: load preprocessed feature files from the directory described by the --features argument.

1) individual feature analysis and 2) combined feature analysis

The combined feature analysis phase uses the results 
from the individual feature analysis to identify important features and group them a combined analysis. 
"""
import argparse
import sys
import dill
import shutil
import time
import os
import gc
from pathos.multiprocessing import cpu_count
from pathos.multiprocessing import ProcessPool as Pool
from fingerprint_modeler import WebsiteFingerprintModeler
from data_utils import WebsiteData, logger
import datetime

# ## GPU Allocation
os.environ["CUDA_VISIBLE_DEVICES"] = "7";


def parse_args():
    """
    Parse command line arguments

    Accepted arguments:
      (f)eatures    -- directory which contains feature files
      (o)output     -- directory where to save analysis results
      n_samples     -- samples for monte-carlo
      nmi_threshold -- redundancy threshold value
      topn          -- number of features to keep after pruning
      n_procs       -- number of processes to use for analysis

    Returns
    -------
    Namespace
        Argument namespace object

    """
    parser = argparse.ArgumentParser("Estimate information leakage for features in website fingerprinting attacks.")

    # Required Arguments
    # directory containing feature files
    parser.add_argument("-f", "--features",
                        required=True,
                        type=str,
                        help="Directory which contains files with the .feature extension.")

    # parser.add_argument("-o", "--output",
    #                     required=True,
    #                     type=str,
    #                     default="/scratch/DA/WeFDE/su/leakage_extracted/v"+ datetime.now().strftime('%y-%m-%d'),
    #                     help="Directory location where to store analysis results.")

    # Optional Arguments
    
    # number of samples for monte-carlo integration
    parser.add_argument("--n_samples",
                        type=int,
                        default=5000,
                        help="The number of samples to use when performing Monte-Carlo Integration estimation. "
                             "Higher values result in more accurate measurements, but longer runtimes.")
    # redundancy threshold
    parser.add_argument("--nmi_threshold",
                        type=float,
                        default=0.9,
                        help="The theshold value used to identify redundant features. "
                             "A value between 0.0 and 1.0.")
    parser.add_argument("--topn",
                        type=int,
                        default=100,
                        help="The number of top features to save during combined feature analysis")
    # number of processes
    parser.add_argument("--n_procs",
                        type=int,
                        default=0,
                        help="The number of processes to use when performing parallel operations. "
                             "Use '0' to use all available processors.")
    parser.add_argument("--discrete_threshold",
                        type=int,
                        default=100000,
                        help="The threshold to use for identifying discrete data samples.")
    return parser.parse_args()


def _individual_measure(modeler, pool, checkpoint):
    """
    Perform information leakage analysis for each feature one-by-one.

    The resulting leakages are saved in a plain-text ascii checkpoint file,
    which can be loaded in subsequent runs to avoid re-processing features.

    Parameters
    ----------
    modeler : WebsiteFingerprintModeler
        initialized fingerprinting engine
    pool : ProcessPool
        Pool to use for multiprocessing.
    checkpoint : str
        Path to ascii file to save individual leakage checkpoint information.

    Returns
    -------
    list
        list of leakages where the index of each leakage maps to the feature number

    """
    leakage_indiv = []

    # open a checkpoint file
    if checkpoint:
        lines = None
        if os.path.exists(checkpoint):
            with open(checkpoint, 'r') as tmp_file:
                past_leaks = [float(line) for line in tmp_file]
                lines = len(past_leaks)
                leakage_indiv = past_leaks
        tmp_file = open(checkpoint, 'a+')

    # if a pool has been provided, perform computation in parallel
    # otherwise do serial computation
    if checkpoint and lines:
        features = modeler.data.features[lines:]
    else:
        features = modeler.data.features
    if pool is None:
        proc_results = map(modeler, features)
    else:
        proc_results = pool.imap(modeler, features)
        pool.close()
    size = len(modeler.data.features)  # number of features

    logger.info("Begin individual leakage measurements.")
    # measure information leakage
    # log current progress at twenty intervals
    for leakage in proc_results:
        leakage_indiv.append(leakage[0])
        if len(leakage_indiv)-1 % int(size*0.05) == 0:
            logger.info("Progress: {}/{}".format(len(leakage_indiv), size))
        if checkpoint:
            tmp_file.write('{}\n'.format(str(leakage[0])))
            tmp_file.flush()
    logger.info("Progress: Done.")
    if pool is not None:
        pool.join()
        pool.restart()
    if checkpoint:
        tmp_file.close()
    return leakage_indiv


def main(features_path, output_path, n_procs=0, n_samples=5000, topn=100, nmi_threshold=0.9, discrete_threshold=100000):
    """
    Run the full information leakage analysis on a processed dataset.

    Parameters
    ----------
    features_path : str
        Operating system file path to the directory containing processed feature files.
    output_path : str
        Operating system file path to the directory where analysis results should be saved.
    n_procs : int
        Number of processes to use for parallelism.
        If 0 is used, auto-detect based on number of system CPUs.
    n_samples : int
        Number of samples to use when performing monte-carlo estimation when running the fingerprint modeler.
    topn : int
        Top number of features to analyze during joint analysis.
    nmi_threshold : float
        Cut-off value for determining redundant features. Should be a percentage value.

    Returns
    -------
    float
        Combined feature leakage (in bits)
    """
    # prepare feature dataset
    print("input_path=", features_path)
    if os.path.exists(output_path): shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    logger.info("Loading dataset.")
    feature_data = WebsiteData(features_path)
    print(feature_data)
    logger.info("Loaded {} sites.".format(len(feature_data.sites)))
    logger.info("Loaded {} instances.".format(len(feature_data)))

    # create process pool
    if n_procs > 1:
        pool = Pool(n_procs)
    elif n_procs == 0:
        pool = Pool(cpu_count())
    else:
        pool = None

    # directory to save results
    outdir = output_path
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # initialize fingerprint modeler
    modeler = WebsiteFingerprintModeler(feature_data, discrete_threshold=discrete_threshold)

    # load previous leakage measurements if possible
    indiv_path = os.path.join(outdir, 'indiv.pkl')
    if os.path.exists(indiv_path):
        with open(indiv_path, "rb") as fi:
            logger.info("Loading individual leakage measures from file.")
            leakage_indiv = dill.load(fi)

    # otherwise do individual measure
    else:
        logger.info("Begin individual feature analysis.")

        # perform individual measure with checkpointing
        chk_path = os.path.join(outdir, 'indiv_checkpoint.txt')
        leakage_indiv = _individual_measure(modeler, pool, chk_path)

        # save individual leakage to file
        logger.info("Saving individual leakage to {}.".format(indiv_path))
        with open(indiv_path, "wb") as fi:
            dill.dump(leakage_indiv, fi)

if __name__ == "__main__":
    try:
        args = parse_args()
        output_path = '/scratch4/kanghosung/starlink_wefde/leakage/' + args.features.split('/')[-1]
        sys.stdout = open(f'/scratch4/kanghosung/starlink_wefde/log/{args.features.split("/")[-1]}.txt', 'w')  
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #/scratch4/kanghosung/starlink_wefde/features/firefox_fiber
        #def main(features_path, output_path, n_procs=0, n_samples=5000, topn=100, nmi_threshold=0.9, discrete_threshold=100000)
        start = time.time()
        main(args.features, output_path,
            n_procs=args.n_procs,
            n_samples=args.n_samples,
            topn=args.topn,
            nmi_threshold=args.nmi_threshold,
            discrete_threshold=args.discrete_threshold)
        end = time.time()
        print("All Done:", (end-start)/60, "min")
        sys.stdout.close()
    except KeyboardInterrupt:
        sys.exit(-1)


