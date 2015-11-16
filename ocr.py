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
    parser.add_argument('--weights', type=str, required=True, help="weight information set to use");
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
    dicer.dump_words('words');

    # shoehorned api - *somebody* doesn't have nice docs, or wrappers, or anything else for that matter
    activedir = str(uuid.uuid4());
    dicer.write_out(activedir);

    # see? a fucking FS pass instead of passing data directly.
    # also system calls. just kill me now.
    print('generating metadata for neural network...');
    os.system('mv out/' + activedir + ' cnn/data/tmp/' + activedir)
    os.system('touch encodedcsv/' + activedir + '.csv');
    receptor.readFolderWithName('tmp/' + activedir, activedir + '.csv', False);

    # now look at this fuckery
    print('initializing network');
    n = network.Network(26,150,7);

    # ugh
    print('loading in generated metadata');
    testingData = network.importCSV('encodedcsv/' + activedir + '.csv');

    # now we're constrained to having the dir named CNN
    print('loading weights...')
    n.importWeights('cnn/weights/' + opts.weights + '/wi.csv', 'cnn/weights/' + opts.weights + '/wo.csv');

    # finally do the recognition
    print('running network...');
    n.recognize(testingData, activedir + '.txt');

    # cuz users are stupid
    print('\n\noutput in ' + activedir + '.txt');



