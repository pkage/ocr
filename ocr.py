#! /usr/bin/env python

from photochop.photochop import Photochopper
from cnn import network, receptor
import argparse, uuid, os


if __name__=="__main__":
    # setting up the argument parser
    parser = argparse.ArgumentParser(description="ocr - the very jankiest of ocr");
    parser.add_argument('filename', type=str, help='the input image file');
    parser.add_argument('--auto-align', action='store_true', required=False, help="auto align the input document");
    parser.add_argument('--pre-smooth', action='store_true', required=False, help="pre-smooth the input document");
    parser.add_argument('--minimum-group-size', type=int, required=False, help="set the minimum group size to be accepted");
    parser.add_argument('--set-threshold-to', type=int, required=False, help="set the threshold for a match (0-255)");
    parser.add_argument('--enable-multiprocessing', action='store_true', default=False, help="Multi-processing, default false");
    parser.add_argument('--read-weights', type=str, required=True, help="weight information set to use");
    opts = parser.parse_args();

    # initialize the argument parser
    dicer = Photochopper(opts.filename, opts.set_threshold_to);

    # if we can set the minimum group size then do that thing
    if opts.minimum_group_size is not None:
        dicer.set_minimum_group_size(opts.minimum_group_size);

    # set some options - only the essentials of course
    dicer.enable_auto_align(opts.auto_align);
    dicer.enable_pre_smoothing(opts.pre_smooth);

    # dicing
    print('chopping the image...');
    dicer.process();
    dicer.process_words();

    # shoehorned api - *somebody* doesn't have nice docs, or wrappers, or anything else for that matter
    activedir = str(uuid.uuid4());
    dicer.write_out(activedir);

    if opts.enable_multiprocessing:
        print("Receptor : Starting Multiprocessing...");
        start = time.time();
        imgs = [];
        try:
            threadcount = cpu_count();
        except NotImplementedError:
            threadcount = 2;
            print('Error getting core counts, setting threadcount to 2');
        threads = Pool(threadcount);

        for x in xrange(1,10):
            imgs.append([arr]); # PATRICK THIS IS WHERE TO INPUT ARR

        print("Distributing tasks across " + str(threadcount) + " cores...");
        final = threads.map(partial(mp), imgs);

        end = time.time();
        print("Time Elapsed: " + str(end - start) + " seconds");

    def mp((arr)):
        receptor = Receptor();
        receptor.setInputArr(arr);
        receptor.generateReceptors();
        receptor.setCharacter("x");
        values = receptor.getOutput();
        return values;
