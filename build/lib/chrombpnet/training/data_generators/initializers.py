import chrombpnet.training.data_generators.batchgen_generator as batchgen_generator
from chrombpnet.training.utils import data_utils
import pandas as pd
import json

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]

def fetch_data_and_model_params_based_on_mode(mode, args, parameters, nonpeak_regions, peak_regions):

    if mode=="train": 
        inputlen=int(parameters["inputlen"])
        outputlen=int(parameters["outputlen"])
        negative_sampling_ratio=float(parameters["negative_sampling_ratio"])
        max_jitter=int(parameters["max_jitter"])
        add_revcomp=True
        shuffle_at_epoch_start=True
        

    elif mode=="valid":
        inputlen=int(parameters["inputlen"])
        outputlen=int(parameters["outputlen"])
        # fix negatives set for validation
        if (nonpeak_regions is not None) and (peak_regions is not None):
            nonpeak_regions=nonpeak_regions.sample(n=int(float(parameters["negative_sampling_ratio"])*peak_regions.shape[0]), replace=False, random_state=args.seed)
        negative_sampling_ratio=1.0 # already subsampled
        # do not jitter at valid time - we are testing only at summits
        max_jitter=0
        # no reverse complementation at valid time
        add_revcomp=False
        # no need to shuffle
        shuffle_at_epoch_start=False

    elif mode=="test":
        # read input/output length
        inputlen=args.inputlen
        outputlen=args.outputlen
        # no subsampling of negatives - test on all positives and negatives
        negative_sampling_ratio=1.0
        # no jitter at valid time - we are testing only at summits
        max_jitter=0
        # no reverse complementation at test time
        add_revcomp=False    
        # no need to shuffle
        shuffle_at_epoch_start=False
        
    else:
        print("mode not defined - only train, valid, test are allowed")

    return inputlen, outputlen,  nonpeak_regions, negative_sampling_ratio, max_jitter, add_revcomp, shuffle_at_epoch_start


def get_bed_regions_for_fold_split(bed_regions, mode, splits_dict):
    chroms_to_keep=splits_dict[mode]
    bed_regions_to_keep=bed_regions[bed_regions["chr"].isin(chroms_to_keep)]
    print("got split:"+str(mode)+" for bed regions:"+str(bed_regions_to_keep.shape))
    return bed_regions_to_keep, chroms_to_keep

def initialize_generators(args, mode, parameters, return_coords):

    # defaults
    peak_regions_atac=None
    nonpeak_regions_atac=None
    peak_regions_dnase=None
    nonpeak_regions_dnase=None


    # get only those peak/non peak regions corresponding to train/valid/test set
    splits_dict=json.load(open(args.chr_fold_path))

    if args.peaks_atac.lower() != "none":
        print("loading peaks...")
        peak_regions_atac=pd.read_csv(args.peaks_atac,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        peak_regions_atac, chroms=get_bed_regions_for_fold_split(peak_regions_atac, mode, splits_dict)

    if args.nonpeaks_atac.lower() != "none":
        print("loading nonpeaks...")
        nonpeak_regions_atac=pd.read_csv(args.nonpeaks_atac,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        nonpeak_regions_atac, chroms=get_bed_regions_for_fold_split(nonpeak_regions_atac, mode, splits_dict)

    if args.peaks_dnase.lower() != "none":
        print("loading peaks...")
        peak_regions_dnase=pd.read_csv(args.peaks_dnase,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        peak_regions_dnase, chroms=get_bed_regions_for_fold_split(peak_regions_dnase, mode, splits_dict)

    if args.nonpeaks_dnase.lower() != "none":
        print("loading nonpeaks...")
        nonpeak_regions_dnase=pd.read_csv(args.nonpeaks_dnase,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        nonpeak_regions_dnase, chroms=get_bed_regions_for_fold_split(nonpeak_regions_dnase, mode, splits_dict)

    inputlen, outputlen, nonpeak_regions_atac, negative_sampling_ratio, max_jitter, add_revcomp, shuffle_at_epoch_start = \
        fetch_data_and_model_params_based_on_mode(mode, args, parameters, nonpeak_regions_atac, peak_regions_atac)
    inputlen, outputlen, nonpeak_regions_dnase, negative_sampling_ratio, max_jitter, add_revcomp, shuffle_at_epoch_start = \
        fetch_data_and_model_params_based_on_mode(mode, args, parameters, nonpeak_regions_dnase, peak_regions_dnase)
    generator=batchgen_generator.ChromBPNetBatchGenerator(
                                    peak_regions_atac=peak_regions_atac,
                                    nonpeak_regions_atac=nonpeak_regions_atac,
                                    peak_regions_dnase=peak_regions_dnase,
                                    nonpeak_regions_dnase=nonpeak_regions_dnase,
                                    genome_fasta=args.genome,
                                    batch_size=args.batch_size,
                                    inputlen=inputlen,                                        
                                    outputlen=outputlen,
                                    max_jitter=max_jitter,
                                    negative_sampling_ratio=negative_sampling_ratio,
                                    cts_bw_file_atac=args.input_bigwig_file_atac,
                                    cts_bw_file_dnase=args.input_bigwig_file_dnase,
                                    add_revcomp=add_revcomp,
                                    return_coords=return_coords,
                                    shuffle_at_epoch_start=shuffle_at_epoch_start
                                    )
    
    return generator
