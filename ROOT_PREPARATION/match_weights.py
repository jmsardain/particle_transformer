import uproot
import awkward as ak
import numpy as np
import glob
import os
import hep_ml.reweight as reweight

def match_weights(pt, target, n_bins=200):
    """ match_weights - This function will use the hepml reweight function
    to calculate weights which match the pt distribution to the target
    distribution. Usually used to match the bkg pt distribution to the
    signal.

    Arguments:
    pt (array) - Distribution to calculate weights for
    target (array) - Distribution to match
    n_bins (int)

    Returns:
    (array) - vector of weights for pt
    """

    # Fit reweighter to target distribution
    reweighter = reweight.BinsReweighter(n_bins=n_bins)
    reweighter.fit(pt, target=target)

    # Predict new weights
    weights = reweighter.predict_weights(pt)
    weights /= weights.mean()

    return weights


def open_sig_and_bkg(file_dir,tree, sig_tag, bkg_tag,step_size=256,replace_files=True):
    #Get Signal File List
    sig_train_files = glob.glob(file_dir+sig_tag+"*train*")
    bkg_train_files = []
    
    #For each signal file, get the bkg file
    for sig_f in sig_train_files:
        #replace sig tag with bkg tag and check file exists
        bkg_f = glob.glob(sig_f.replace(sig_tag,bkg_tag))
        if len(bkg_f) != 1: raise Exception("bkg and sig must have matching file numbers")
        else: bkg_train_files.append(bkg_f[0])

    #For each file, get the jet_pt, match the weights and write the file
    for sig_f,bkg_f in zip(sig_train_files,bkg_train_files):
        print("Matching weights of ",bkg_f,"and",sig_f+"...",end="",flush=True)

        with uproot.open(sig_f+tree) as sig:
            sig_pt = sig["jet_pt"].array()
        
        with uproot.open(bkg_f+tree) as bkg:
            bkg_pt = bkg["jet_pt"].array()
            bkg_weight=match_weights(bkg_pt,sig_pt) #use sig_pt to reweight bkg
        
        #Iterate through bkg file, set weight and write
        out_file = None
        count = 0
        for array in uproot.iterate(bkg_f+tree, step_size=step_size):
            
            #Open output file on first iteration
            if count == 0:
                branches = {}
                for column in array.fields:
                    # can only write out 2D awkward arrays annoyingly
                    if not array[column].ndim > 2:
                        branches[column] = ak.type(array[column]) #get types for each field
                
                out_file = uproot.recreate(bkg_f.replace(".root",".part.root"))
                out_file.mktree("tree", branches)
            
            #Set weight
            if count+step_size<len(bkg_weight):
                array["weight"] = bkg_weight[count:count+step_size]
                count+=step_size
            else:
                array["weight"] = bkg_weight[count:]
            
            #Writes to file
            out_file["tree"].extend(array)
        out_file.close()
        
        if replace_files:
            os.replace(bkg_f.replace(".root",".part.root"),bkg_f)
        print("Completed!")            



open_sig_and_bkg("data_out/",":tree","sig","bkg",step_size=256,replace_files=True)    
