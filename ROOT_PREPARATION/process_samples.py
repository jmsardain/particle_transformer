import uproot
import awkward as ak
import numpy as np
import glob
from prep_functions import common_cuts, signal_cuts, simple_angular

def selection_cuts(batch, signal_bool):
    #Apply selections cuts depending on sig/bkg
    cuts = signal_cuts(batch) if signal_bool else common_cuts(batch)
    batch = {kw: batch[kw][cuts,...] for kw in batch.fields}
    return batch
    
def preprocess(jets,branches):
    #sort constits by pt
    pt=jets["fjet_clus_pt"]
    pt_pad = ak.pad_none(pt,3, axis=1)
    pt_pad = ak.fill_none(pt_pad, 0, axis=1)
    sort_indeces = ak.argsort(pt, axis=1,ascending=False) #decreasing pt

    #centre jets using 3 lead jets
    jets = simple_angular(
        jets,
        sort_indeces
        )

    #sort by decreasing pt
    for branch in branches:
        if "_clus_" in branch:
            jets[branch] = jets[branch][sort_indeces]
        
    return jets

def split_input_files(config):
    max_jets = config["max_jets"]
    num_outputs = config["num_outputs"]
    #get filenames
    filenames = glob.glob(config["in_file"])
    jet_count_dict = dict((fn,0) for fn in filenames)
    
    #Find the number of jets in each file
    for fn in filenames:
        for _data in uproot.iterate(fn+config['tree'], filter_name = config["branches_to_use_in_preprocess"]):
            _data = selection_cuts(_data,config["signal"]) #Apply selection cuts to avoid later problems with jets being cut
            jet_count_dict[fn]+=len(_data[config["branches_to_use_in_preprocess"][0]])

    num_jets_in_samples = sum(jet_count_dict.values())
    
    if max_jets is None: max_jets = num_jets_in_samples #if None, take all jets
    
    #ensure max_jets is smaller or equal than num_jets_in_sample
    if max_jets > num_jets_in_samples:
        print("\nWARNING: max_jets too high. Instead saving max number of jets in sample:",num_jets_in_samples,"\n")
        max_jets = num_jets_in_samples
        
    #Take fraction of jets determined by max_jets
    for key in jet_count_dict.keys():
        jet_count_dict[key] = int(jet_count_dict[key]*max_jets/num_jets_in_samples) #Round down
    len_output = np.floor(max_jets/num_outputs)
        
    print("Processing into",num_outputs,"output files: Each with length",len_output,"Jets")
    print("TOTAL JETS:",int(num_outputs*len_output)) #Number of jets may have been rounded down
    
    return jet_count_dict, len_output

def skim(in_file_name, out_file_names, signal, branches_to_use_in_preprocess, branches_to_keep, max_constits=80, max_jets=None, out_file=None, output_length=None,out_file_index=0,num_jets_file = 0):
    num_jets_total = 0
    
    # read
    for input_data in uproot.iterate(in_file_name, filter_name = branches_to_use_in_preprocess):

        #Define other training quantities
        input_data["label_QCD"]=not signal
        input_data["label_sig"]=signal
        input_data["jet_pt"] = np.sum(input_data["fjet_clus_pt"],axis=-1)
        input_data["jet_energy"] = np.sum(input_data["fjet_clus_E"],axis=-1)
        input_data["weight"]=1.0 #SET WEIGHTS TO 1, USE MATCH_WEIGHTS.PY TO CHANGE THIS LATER

        #Preprocess and selection cuts
        input_data = preprocess(input_data,branches_to_keep)
        input_data = selection_cuts(input_data,signal) #CUTTING AWAY EVENTS MAY CAUSE PROBLEMS IF MAX_JETS IS TOO HIGH

        #ADD EVENTS UNTIL MAX_JETS PASSED
        if not (max_jets is None):
            new_num_jets = num_jets_total+len(input_data["fjet_clus_pt"])
            if  new_num_jets < max_jets:
                num_jets_total = new_num_jets
            else:
                for branch in branches_to_keep:
                    input_data[branch] = input_data[branch][:max_jets-num_jets_total]
                num_jets_total=max_jets
        
        for branch in branches_to_keep:
            if "_clus_" in branch:
                #PAD and CLIP
                input_data[branch] = ak.pad_none(input_data[branch],max_constits,axis=1,clip=True)
                input_data[branch] = ak.fill_none(input_data[branch],-999,axis=1)
                input_data[branch] = ak.values_astype(input_data[branch], "float32")
                       
        branches = {}
        for column in input_data.keys():
            # can only write out 2D awkward arrays annoyingly
            if not input_data[column].ndim > 2 and column in branches_to_keep:
                branches[column] = ak.type(input_data[column])
                       
        #Open Target Files
        if out_file is None:
            out_file = []
            for fn in out_file_names:
                out = uproot.recreate(fn)
                out.mktree("tree", branches)    
                out_file.append(out)
        
        #Write Jets
        branch_splits = {branch:np.array_split(np.array(input_data[branch]), len(out_file)) for branch in list(branches.keys())}
        print("Writing",len(input_data["fjet_clus_pt"]),"Jets to target files...",end="")
        for i in range(len(out_file)):
            out_file[i]["tree"].extend({branch: branch_splits[branch][i] for branch in branches.keys()})
        print("Done")
        #out_file,out_file_index,num_jets_file = write(input_data, out_file_names, out_file, branches, output_length, num_jets_file,out_file_index)
            
        if num_jets_total == max_jets: break #If max jets from input file reached, stop uproot.iterate(...)
    return out_file, out_file_index, num_jets_file

def run(config):
    #create output file names
    out_file_names = [config["out_file_dir"]+config["out_file_tag"]+"_"+("test" if i < config["num_test"] else "train")+"_"+str(i)+".root" for i in range(config["num_outputs"])]

    #get number of jets to save per input file and the number to save per output file
    jet_count_dict, output_length = split_input_files(config)
    
    out_file = None
    out_file_index = 0
    num_jets_file = 0

    #Skim each input file
    for key in jet_count_dict.keys():
        print("Loading file:",key)
        out_file,out_file_index, num_jets_file = skim(key+config["tree"], out_file_names, config["signal"], config["branches_to_use_in_preprocess"], config["branches_to_keep"], config["max_constits"], jet_count_dict[key], out_file=out_file,output_length=output_length,out_file_index=out_file_index,num_jets_file=num_jets_file)

    for f in out_file:
        f.close()
    print("\nNOTE: WEIGHTS HAVE BEEN ALL SET TO 1. USE MATCH_WEIGHTS.PY TO MATCH WEIGHTS OF SIG AND BKG USING PT.")
        

    
