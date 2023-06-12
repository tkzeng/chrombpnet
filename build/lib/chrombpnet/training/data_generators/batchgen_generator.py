from tensorflow import keras
from chrombpnet.training.utils import augment
from chrombpnet.training.utils import data_utils
import tensorflow as tf
import numpy as np
import random
import string
import math
import os
import json

def subsample_nonpeak_data(nonpeak_seqs, nonpeak_cts_atac, nonpeak_cts_dnase, nonpeak_coords, peak_data_size, negative_sampling_ratio):
    #Randomly samples a portion of the non-peak data to use in training
    num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
    nonpeak_indices_to_keep = np.random.choice(len(nonpeak_seqs), size=num_nonpeak_samples, replace=False)
    nonpeak_seqs = nonpeak_seqs[nonpeak_indices_to_keep]
    nonpeak_cts_atac = nonpeak_cts_atac[nonpeak_indices_to_keep]
    nonpeak_cts_dnase = nonpeak_cts_dnase[nonpeak_indices_to_keep]
    nonpeak_coords = nonpeak_coords[nonpeak_indices_to_keep]
    return nonpeak_seqs, nonpeak_cts_atac, nonpeak_cts_dnase, nonpeak_coords

class ChromBPNetBatchGenerator(keras.utils.Sequence):
    """
    This generator randomly crops (=jitter) and revcomps training examples for 
    every epoch, and calls bias model on it, whose outputs (bias profile logits 
    and bias logcounts) are fed as input to the chrombpnet model.
    """
    def __init__(self, peak_regions_atac, nonpeak_regions_atac, peak_regions_dnase, nonpeak_regions_dnase,
                 genome_fasta, batch_size, inputlen, outputlen, max_jitter, negative_sampling_ratio,
                 cts_bw_file_atac, cts_bw_file_dnase,
                 add_revcomp, return_coords, shuffle_at_epoch_start):
        """
        seqs: B x L' x 4
        cts: B x M'
        inputlen: int (L <= L'), L' is greater to allow for cropping (= jittering)
        outputlen: int (M <= M'), M' is greater to allow for cropping (= jittering)
        batch_size: int (B)
        """
        # outputlen = inputlen for atac
        # atac coords, atac seq
        peak_seqs_atac, peak_cts_atac_atac, peak_coords_atac, \
        nonpeak_seqs_atac, nonpeak_cts_atac_atac, nonpeak_coords_atac, = \
            data_utils.load_data(peak_regions_atac, nonpeak_regions_atac, genome_fasta,
                                 cts_bw_file_atac, inputlen, inputlen, max_jitter)
        # outputlen = inputlen for atac
        # dnase coords, atac seq
        peak_seqs_dnase, peak_cts_dnase_atac, peak_coords_dnase, \
        nonpeak_seqs_dnase, nonpeak_cts_dnase_atac, nonpeak_coords_dnase, = \
            data_utils.load_data(peak_regions_dnase, nonpeak_regions_dnase, genome_fasta,
                                 cts_bw_file_atac, inputlen, inputlen, max_jitter)

        # atac coords, dnase seq
        _, peak_cts_atac_dnase, _, \
        _, nonpeak_cts_atac_dnase, _, = \
            data_utils.load_data(peak_regions_atac, nonpeak_regions_atac, genome_fasta,
                                 cts_bw_file_dnase, inputlen, outputlen, max_jitter)

        # dnase coords, dnase seq
        _, peak_cts_dnase_dnase, _, \
        _, nonpeak_cts_dnase_dnase, _, = \
            data_utils.load_data(peak_regions_dnase, nonpeak_regions_dnase, genome_fasta,
                                 cts_bw_file_dnase, inputlen, outputlen, max_jitter)

        self.peak_seqs_atac, self.nonpeak_seqs_atac = peak_seqs_atac, nonpeak_seqs_atac
        self.peak_cts_atac_atac, self.nonpeak_cts_atac_atac = peak_cts_atac_atac, nonpeak_cts_atac_atac
        self.peak_cts_dnase_atac, self.nonpeak_cts_dnase_atac = peak_cts_dnase_atac, nonpeak_cts_dnase_atac
        self.peak_coords_atac, self.nonpeak_coords_atac = peak_coords_atac, nonpeak_coords_atac

        self.peak_seqs_dnase, self.nonpeak_seqs_dnase = peak_seqs_dnase, nonpeak_seqs_dnase
        self.peak_cts_atac_dnase, self.nonpeak_cts_atac_dnase = peak_cts_atac_dnase, nonpeak_cts_atac_dnase
        self.peak_cts_dnase_dnase, self.nonpeak_cts_dnase_dnase = peak_cts_dnase_dnase, nonpeak_cts_dnase_dnase
        self.peak_coords_dnase, self.nonpeak_coords_dnase = peak_coords_dnase, nonpeak_coords_dnase

        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.batch_size = batch_size
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start


        # random crop training data to the desired sizes, revcomp augmentation
        self.crop_revcomp_data()

    def __len__(self):

        return math.ceil(self.seqs.shape[0]/self.batch_size)


    def crop_revcomp_data(self):
        # random crop training data to inputlen and outputlen (with corresponding offsets), revcomp augmentation
        # shuffle required since otherwise peaks and nonpeaks will be together
        #Sample a fraction of the negative samples according to the specified ratio
        #if (self.peak_seqs_atac is not None) and (self.nonpeak_seqs_atac is not None):

        # crop peak data before stacking
        peaks_cts_atac = np.vstack([self.peak_cts_atac_atac,self.peak_cts_dnase_atac])
        peaks_cts_dnase = np.vstack([self.peak_cts_atac_dnase, self.peak_cts_dnase_dnase])
        nonpeak_cts_atac = np.vstack([self.nonpeak_cts_atac_atac,self.nonpeak_cts_dnase_atac])
        nonpeak_cts_dnase = np.vstack([self.nonpeak_cts_atac_dnase, self.nonpeak_cts_dnase_dnase])
        self.peak_seqs = np.vstack([self.peak_seqs_atac, self.peak_seqs_dnase])
        self.nonpeak_seqs = np.vstack([self.nonpeak_seqs_atac, self.nonpeak_seqs_dnase])
        self.peak_coords = np.vstack([self.peak_coords_atac, self.peak_coords_dnase])
        self.nonpeak_coords = np.vstack([self.nonpeak_coords_atac, self.nonpeak_coords_dnase])
        # print("peaks_cts_atac", peaks_cts_atac.shape, "self.peak_seqs_atac", self.peak_seqs_atac.shape)
        # print("peaks_cts_dnase", peaks_cts_dnase.shape, "self.peak_seqs_dnase", self.peak_seqs_dnase.shape)
        # print("nonpeaks_cts_atac", nonpeak_cts_atac.shape, "self.nonpeak_seqs_atac", self.nonpeak_seqs_atac.shape)
        # print("nonpeaks_cts_dnase", nonpeak_cts_dnase.shape, "self.nonpeak_seqs_dnase", self.nonpeak_seqs_dnase.shape)
        # only crop output (dnase) to 1kb
        cropped_peaks, cropped_cnts_atac, cropped_cnts_dnase, cropped_coords = \
            augment.random_crop(self.peak_seqs, peaks_cts_atac, peaks_cts_dnase,
                                self.inputlen, self.outputlen,
                                self.peak_coords)
        #cropped_peaks_dnase, cropped_cnts_dnase, cropped_coords_dnase = \
        #    augment.random_crop(self.peak_seqs_dnase, peaks_cts_dnase, self.inputlen, self.outputlen, self.peak_coords_dnase)

        if self.negative_sampling_ratio < 1.0:
            self.sampled_nonpeak_seqs, self.sampled_nonpeak_cts_atac, self.sampled_nonpeak_cts_dnase, self.sampled_nonpeak_coords = \
                subsample_nonpeak_data(self.nonpeak_seqs, nonpeak_cts_atac, nonpeak_cts_dnase, self.nonpeak_coords,
                                       len(self.peak_seqs), self.negative_sampling_ratio)
            # self.seqs_atac = np.vstack([cropped_peaks_atac, self.sampled_nonpeak_seqs_atac])
            # self.cts_atac_atac = np.vstack([cropped_cnts_atac_atac, self.sampled_nonpeak_cts_atac_atac])
            # self.cts_atac_dnase = np.vstack([cropped_cnts_atac_dnase, self.sampled_nonpeak_cts_atac_dnase])
            # self.coords_atac = np.vstack([cropped_coords_atac, self.sampled_nonpeak_coords_atac])
            #
            # self.seqs_dnase = np.vstack([cropped_peaks_dnase, self.sampled_nonpeak_seqs_dnase])
            # self.cts_dnase_atac = np.vstack([cropped_cnts_dnase_atac, self.sampled_nonpeak_cts_dnase_atac])
            # self.cts_dnase_dnase = np.vstack([cropped_cnts_dnase_dnase, self.sampled_nonpeak_cts_dnase_dnase])
            # self.coords_dnase = np.vstack([cropped_coords_dnase, self.sampled_nonpeak_coords_dnase])

            self.seqs = np.vstack([cropped_peaks, self.sampled_nonpeak_seqs])
            self.cts_atac = np.vstack([cropped_cnts_atac, self.sampled_nonpeak_cts_atac])
            self.cts_dnase = np.vstack([cropped_cnts_dnase, self.sampled_nonpeak_cts_dnase])
            self.coords = np.vstack([cropped_coords, self.sampled_nonpeak_coords])
        else:
            self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
            self.cts_atac = np.vstack([cropped_cnts_atac, nonpeak_cts_atac])
            self.cts_dnase = np.vstack([cropped_cnts_dnase, nonpeak_cts_dnase])
            self.coords = np.vstack([cropped_coords, self.nonpeak_coords])


        # elif self.peak_seqs is not None:
        #     # crop peak data before stacking
        #     cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords)
        #
        #     self.seqs = cropped_peaks
        #     self.cts = cropped_cnts
        #     self.coords = cropped_coords
        #
        # elif self.nonpeak_seqs is not None:
        #     #print(self.nonpeak_seqs.shape)
        #
        #     self.seqs = self.nonpeak_seqs
        #     self.cts = self.nonpeak_cts
        #     self.coords = self.nonpeak_coords
        # else :
        #     print("Both peak and non-peak arrays are empty")


        # self.cur_seqs_atac = self.seqs_atac
        # self.cur_cts_atac_atac = self.cts_atac_atac
        # self.cur_cts_atac_dnase = self.cts_atac_dnase
        # self.cur_coords_atac = self.coords_atac
        #
        # self.cur_seqs_dnase = self.seqs_dnase
        # self.cur_cts_dnase_atac = self.cts_dnase_atac
        # self.cur_cts_dnase_dnase = self.cts_dnase_dnase
        # self.cur_coords_dnase = self.coords_dnase

        self.cur_seqs = self.seqs
        self.cur_cts_atac = self.cts_atac
        self.cur_cts_dnase = self.cts_dnase
        self.cur_coords = self.coords

        # todo: implement crop_revcomp_augment
        # self.cur_seqs_atac, self.cur_cts_atac_atac, self.cur_cts_atac_dnase, self.cur_coords_atac = augment.crop_revcomp_augment(
        #     self.seqs_atac, self.cts_atac_atac, self.cts_atac_dnase, self.coords_atac,
        #     self.inputlen, self.outputlen, self.add_revcomp, shuffle=self.shuffle_at_epoch_start
        # )
        # self.cur_seqs_dnase, self.cur_cts_dnase_atac, self.cur_cts_dnase_dnase, self.cur_coords_dnase = augment.crop_revcomp_augment(
        #     self.seqs_atac, self.cts_dnase_atac, self.cts_dnase_dnase, self.coords_dnase,
        #     self.inputlen, self.outputlen, self.add_revcomp, shuffle=self.shuffle_at_epoch_start
        # )

    def __getitem__(self, idx):
        batch_seq = self.cur_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_cts_atac = self.cur_cts_atac[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_cts_dnase = self.cur_cts_dnase[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_coords = self.cur_coords[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_cts_atac = np.expand_dims(batch_cts_atac, axis=-1)

        # print("seq",batch_seq)
        # print("cts atac", batch_cts_atac)
        # print("cts dnase", batch_cts_dnase)
        # print("summed cts dnase", np.log(1+batch_cts_dnase.sum(-1, keepdims=True)))

        if self.return_coords:
            return (np.concatenate([batch_seq,batch_cts_atac], axis=-1),
                    [batch_cts_dnase, np.log(1+batch_cts_dnase.sum(-1, keepdims=True))],
                    batch_coords)
        else:
            # np.log(1+batch_cts_atac.sum(-1, keepdims=True))
            return (np.concatenate([batch_seq,batch_cts_atac], axis=-1),
                    [batch_cts_dnase,np.log(1+batch_cts_dnase.sum(-1, keepdims=True))])

    def on_epoch_end(self):
        pass #todo: implement
        #self.crop_revcomp_data()

