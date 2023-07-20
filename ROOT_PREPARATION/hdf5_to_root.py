#THIS DOES NOT YET WORK!!!!!!

import h5py
import uproot
import awkward as ak
import numpy as np

in_file = "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/ConstituentTopTagging/Root2HDF5/dataloc/test_s2_ln.h5"
out_file = "temp"

h5 = h5py.File(in_file, 'r')

step = 10000
for i in range(0,10000,step):
    h5chunk = h5["constit"][i:i+step,:,:]
    pt=np.exp(h5chunk[:,:,2])
    labels = h5["labels"][i:i+step]
    
    print(labels)
    print(np.logical_not(labels))
    
    chunk_dict = {"part_deta":h5chunk[:,:,0],"part_dphi":h5chunk[:,:,1],"part_energy":np.exp(h5chunk[:,:,3]),
                  "part_px":pt*np.cos(h5chunk[:,:,1]),"part_py":pt*np.sin(h5chunk[:,:,1]),"part_pz":pt*np.sinh(h5chunk[:,:,0]),
                  "label_QCD":np.logical_not(labels),"label_sig":labels,"jet_pt":np.sum(pt,axis=-1)}
    output = uproot.recreate(out_file+str(int(i/step))+".root")
    output["tree"] = chunk_dict

    print(i)

#THERE WILL BE A LOT OF ZEROS IF NUM_CONSTITS < MAX_CONSTITS (200)

h5.close()