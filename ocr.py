#! /usr/bin/env python

from photochop.photochop import Photochopper
from cnn import network, receptor
import argparse, uuid, os, time, enchant
from multiprocessing import Pool, cpu_count
from functools import partial



def mp(arr):
    rec = receptor.Receptor();
    rec.setInputArr(arr);
    rec.generateReceptors();
    rec.setCharacter("x");
    values = rec.getOutput();
    return values;

if __name__=="__main__":
    # setting up the argument parser
    parser = argparse.ArgumentParser(description="ocr - the very jankiest of ocr");
    parser.add_argument('filename', type=str, help='the input image file');
    parser.add_argument('--auto-align', action='store_true', required=False, help="auto align the input document");
    parser.add_argument('--pre-smooth', action='store_true', required=False, help="pre-smooth the input document");
    parser.add_argument('--minimum-group-size', type=int, required=False, help="set the minimum group size to be accepted");
    parser.add_argument('--set-threshold-to', type=int, required=False, help="set the threshold for a match (0-255)", default=200);
    parser.add_argument('--read-weights', type=str, required=True, help="weight information set to use");
    parser.add_argument('--spellcheck', action='store_true', required=False, help='spellcheck everything');
    opts = parser.parse_args();

    opts.enable_multiprocessing = True;

    if opts.spellcheck:
        chk = enchant.Dict('en_US');

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

    print('initializing neural network...');
    net = network.Network(26, 150, 7);
    net.importWeights(os.path.join('cnn', 'weights', opts.read_weights, 'wi.csv'), os.path.join('cnn', 'weights', opts.read_weights, 'wo.csv'));

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

        doc = [];
        
        for key in dicer.words:
            line = dicer.words[key];
            linedata = "";
            
            for i in range(0, len(line)):
                word = line[i];
                imgs = [];
                for img in word:
                    imgs.append(img);
                final = threads.map(partial(mp), imgs);
                net.setTestingData(final);
                net.outlist = [];
                net.recognize();
                wrd = ''.join(net.outlist);
                if opts.spellcheck and not chk.check(wrd) and (i + 1 != len(line)) and i != 0:
                    sgg = chk.suggest(wrd); 
                    wrd = sgg[0] if len(sgg) > 0 else wrd;
                linedata += wrd + " "; 
                
            doc.append(linedata);

        print('output:\n' + '\n'.join(['\t' + line for line in doc]));
                    

                
            

    #    print("Distributing tasks across " + str(threadcount) + " cores...");
        final = threads.map(partial(mp), imgs);

        end = time.time();
        print("Time Elapsed: " + str(end - start) + " seconds");

