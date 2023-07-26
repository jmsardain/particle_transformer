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

def shuffle_and_merge(file_dir,tree,sig_tag,bkg_tag):
    #Get Signal File List
    sig_file_names = glob.glob(file_dir+sig_tag+"*")
    bkg_file_names = []
    test_file = None
    train_file = None
    #For each signal file, get the bkg file
    for sig_f in sig_file_names:
        #replace sig tag with bkg tag and check file exists
        bkg_f = glob.glob(sig_f.replace(sig_tag,bkg_tag))
        if len(bkg_f) != 1: raise Exception("bkg and sig must have matching file numbers")
        else: bkg_file_names.append(bkg_f[0])

    #For each file, get the jet_pt, match the weights and write the file
    for sig_f,bkg_f in zip(sig_file_names,bkg_file_names):
        print("Combining ",bkg_f,"and",sig_f+"...",end="",flush=True)

        #Combine into single root file
        array = uproot.concatenate({sig_f:tree,bkg_f:tree})
        indices = np.arange(len(array))
        
        #Shuffle sig and bkg
        seed=0
        rng = np.random.default_rng(seed)
        rng.shuffle(indices, axis=0)
                 
        #Open output files on first iteration
        if test_file == None:
            branches = {}
            for column in array.fields:
                # can only write out 2D awkward arrays annoyingly
                if not array[column].ndim > 2:
                    branches[column] = ak.type(array[column]) #get types for each field
            
            #open temporary files
            test_file = uproot.recreate("test.part.root")
            test_file.mktree("tree", branches)
            train_file = uproot.recreate("train.part.root")
            train_file.mktree("tree", branches)
        
        #Save to file
        if "test" in sig_f:
            test_file["tree"].extend(array[indices])
        else:
            train_file["tree"].extend(array[indices])
        
        print("Completed!")
    train_file.close()
    test_file.close()
    
def open_sig_and_bkg(file_dir,tree, sig_tag, bkg_tag,step_size=256):
       
    #Shuffle sig/bkg and merge files            
    shuffle_and_merge(file_dir,tree, sig_tag, bkg_tag)
    
    print("TEMPORARY TEST/TRAIN FILES CREATED. LOOPING THROUGH TRAIN TO MATCH BKG WEIGHTS TO SIG.")
    #ONLY NEED TO REWEIGHT TRAIN FILE
    with uproot.open("train.part.root"+":"+tree) as sig:
        pt = sig["jet_pt"].array()
        labels = sig["label_sig"].array()
        
        bkg_weight=match_weights(pt[labels==0],pt[labels==1]) #use sig pt to reweight bkg
            
    #Iterate through bkg file, set weight and write
    out_file = None
    count = 0
    for array in uproot.iterate("train.part.root"+":"+tree, step_size=step_size):
        
        #Open output file on first iteration
        if count == 0:
            branches = {}
            for column in array.fields:
                # can only write out 2D awkward arrays annoyingly
                if not array[column].ndim > 2:
                    branches[column] = ak.type(array[column]) #get types for each field
            
            out_file = uproot.recreate("train.root")
            out_file.mktree("tree", branches)
        
        #Set weight
        n_bkg = len(array["weight"][array["label_sig"]==0]) #How many bkg are in this chunk
        indeces = np.arange(0, len(array["weight"]), 1)
        bkg_ind = indeces[array["label_sig"] == 0]
        weights = np.ones(len(array["weight"]), dtype=np.float32)
        if count+n_bkg<len(bkg_weight):
            weights[bkg_ind] = bkg_weight[count:count+n_bkg]
            #array = ak.where(array["label_sig"]==0, bkg_weight[count:count+n_bkg], array["weight"])
            array["weight"] = weights
            count+=n_bkg
        else:
            weights[bkg_ind] = bkg_weight[count:]
            #array = ak.where(array["label_sig"]==0, bkg_weight[count:], array["weight"])
            array["weight"] = weights
        
        #Writes to file
        out_file["tree"].extend(array)
    out_file.close()
    
    print("Training file reweighted. Replacing temporary files.")
    
    os.remove("train.part.root")
    os.replace("test.part.root","test.root")
    
    print("\nCOMPLETED!")

open_sig_and_bkg("data_out/","tree","sig","bkg",step_size=256)    
