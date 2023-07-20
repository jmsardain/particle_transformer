import uproot
import awkward as ak
import numpy as np

in_file = "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/ntuples/user.almay.32814930._000001.tree.root"
tree = "FlatSubstructureJetTree"
out_file_name = "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/data_out/sig_1.root"

branches_to_keep = ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi","fjet_clus_eta","label_QCD","label_sig","jet_pt","jet_energy"]
branches_to_use_in_preprocess = ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi","fjet_clus_eta", 'fjet_truthJet_eta','fjet_truthJet_pt',
                                 'fjet_numConstituents','fjet_m','fjet_truth_dRmatched_particle_flavor','fjet_truth_dRmatched_particle_dR',
                                 'fjet_truthJet_dRmatched_particle_dR_top_W_matched','fjet_ungroomed_truthJet_m','fjet_ungroomed_truthJet_Split23',
                                 'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount','fjet_ungroomed_truthJet_pt']
output_branches=[]

signal = True
max_constits = 80
max_events = int(2e5)

def common_cuts(batch):
    """ common_cuts - This function will take in a batch of data (almost always as loaded)
    by uproot.iterate and apply the common cuts for Rel22. For when data format does not
    allow uproot to do this for us.

    Arguments:
    batch (obj or dict) - The batch, where branches are accessible by string names
    exit_check (list of strings) - The names of branches we wish to check for -999 exit codes

    Returns:
    (array) - A boolean array of len branch.shape[0]. If True, jet passes common cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(abs(batch['fjet_truthJet_eta']) < 2.0)
    cuts.append(batch['fjet_truthJet_pt'] / 1000. > 350.)
    cuts.append(batch['fjet_numConstituents'] >= 3)
    cuts.append(batch['fjet_m'] / 1000. > 40.)

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts


def signal_cuts(batch):
    """ signal_cuts - Calls the above function to produce the common cuts, but
    also adds a set of signal cuts which should be applied to the Z' sample.

    Arguments:
    batch (obj or dict) - The batch data from which to compute cuts

    Returns:
    (array) - Boolean array representing total cuts
    """

    # Assemble boolean arrays
    cuts = []
    cuts.append(common_cuts(batch))
    cuts.append(abs(batch['fjet_truth_dRmatched_particle_flavor']) == 6)
    cuts.append(abs(batch['fjet_truth_dRmatched_particle_dR']) < 0.75)
    cuts.append(abs(batch['fjet_truthJet_dRmatched_particle_dR_top_W_matched']) < 0.75)
    cuts.append(batch['fjet_ungroomed_truthJet_m'] / 1000. > 140.)
    cuts.append(batch['fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount'] >= 1)
    cuts.append(batch['fjet_ungroomed_truthJet_Split23'] / 1000. > np.exp(3.3-6.98e-4*batch['fjet_ungroomed_truthJet_pt']/1000.))

    # Take and of all cuts
    total_cuts = np.logical_and.reduce(cuts)

    return total_cuts

def simple_angular(jets, sort_indeces, max_constits=200, **kwargs):
    """ simple_angular - This function will perform a simple preprocessing
    on the angular constituent information. It is the same preprocessing used
    in 1902.09914.

    Arguments:
    jets (dict): Dictionary whose elements are awkard arrays giving the constituent
    eta and phi. Usually a batch of a loop over a .root file using uproot.
    sort_indeces (array): The indeces which will sort the constituents by INCREASING pt. Sort
    will be reflected to sort by decreasing pt.
    zero_indeces (array): The indeces of constituents which do not pass
    minimum pT cut. These are masked to zero.
    max_constits (int): The number of constituents to keep in our jets. Jets shorter
    than this will be zero padded, jets longer than this will be truncated.

    Returns:
    (array) - A zero padded array giving the preprocessed eta values. It's shape will
    be (num_jets, max_constits)
    (array) - A zero padded array giving the preprocessed phi values.
    """

    # Need to center/rotate/flip constituents BEFORE zero padding.
    eta = jets['fjet_clus_eta']
    phi = jets['fjet_clus_phi']

    # 1. Center hardest constituent in eta/phi plane
    # Find the eta/phi coordinates of hardest constituent in each jet, going to
    # need some fancy indexing
    ax_index = np.arange(0, len(eta), 1)
    first_eta = eta[ax_index, sort_indeces[:,0]]
    first_phi = phi[ax_index, sort_indeces[:,0]]

    # Now center
    eta_center = eta - first_eta[:,np.newaxis]
    phi_center = phi - first_phi[:,np.newaxis]

    # Fix discontinuity in phi at +-pi
    phi_center = np.where(phi_center > np.pi, phi_center - 2*np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2*np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    # Screen indeces for any jets with 1 or 2 constituents (ask about these)
    sort_indeces = ak.fill_none(ak.pad_none(sort_indeces,3, axis=1),0,axis=1)

    second_eta = eta_center[ax_index, sort_indeces[:,1]]
    second_phi = phi_center[ax_index, sort_indeces[:,1]]
    angle = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = eta_center * np.cos(angle[:,np.newaxis]) + phi_center * np.sin(angle[:,np.newaxis])
    phi_rot = -eta_center * np.sin(angle[:,np.newaxis]) + phi_center * np.cos(angle[:,np.newaxis])

    # 3. If needed, reflect 3rd hardest constituent into positive eta half-plane
    third_eta = eta_rot[ax_index, sort_indeces[:,2]]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = eta_rot * parity[:,np.newaxis]

    # Finished preprocessing. Return results
    return eta_flip, phi_rot

def selection_cuts(batch):
    cuts = signal_cuts(batch) if signal else common_cuts(batch)
    batch = {kw: batch[kw][cuts,...] for kw in batch.fields}
    return batch
    
def preprocess(jets,branches):
    pt=jets["fjet_clus_pt"]
    pt_pad = ak.pad_none(pt,3, axis=1)
    pt_pad = ak.fill_none(pt_pad, 0, axis=1)
    sort_indeces = ak.argsort(pt, axis=1,ascending=False) #decreasing pt

    eta, phi = simple_angular(
        jets,
        sort_indeces,
        max_constits=max_constits
        )
    
    jets["fjet_clus_eta"]=eta
    jets["fjet_clus_phi"]=phi
    
    #sort by decreasing pt
    for branch in branches:
        if "_clus_" in branch:
            jets[branch] = jets[branch][sort_indeces]
        
    return jets

def skim():
    num_events = 0
    out_file = None
    # read
    for input_data in uproot.iterate(f"{in_file}:{tree}", filter_name = branches_to_use_in_preprocess):
        
        input_data["label_QCD"]=not signal
        input_data["label_sig"]=signal
        input_data["jet_pt"] = np.sum(input_data["fjet_clus_pt"],axis=-1)
        input_data["jet_energy"] = np.sum(input_data["fjet_clus_E"],axis=-1)
        
        # register datatypes to dict
        branches = {}
        for column in input_data.fields:
            # can only write out 2D awkward arrays annoyingly
            if not input_data[column].ndim > 2 and column in branches_to_keep:
                branches[column] = ak.type(input_data[column])
                
        input_data = preprocess(input_data,branches_to_keep)
        input_data = selection_cuts(input_data)
        
        #ADD EVENTS UNTIL MAX_EVENTS PASSED
        if num_events+len(input_data["fjet_clus_pt"]) < max_events:
            num_events += len(input_data["fjet_clus_pt"])
        else:
            for branch in branches_to_keep:
                input_data[branch] = input_data[branch][:max_events-num_events]
            num_events=max_events

        constit_length = ak.count_nonzero(input_data["fjet_clus_pt"],axis=1)
        for branch in branches_to_keep:
            if "_clus_" in branch:
                #Keep fewer than max_constits constituents
                input_data[branch] = np.where(constit_length<=max_constits,input_data[branch],input_data[branch][:,:max_constits-1])
                
        # write
        if out_file is None:
            out_file = uproot.recreate(out_file_name)
            out_file.mktree("tree", branches)
            
        out_file["tree"].extend({branch: input_data[branch] for branch in branches.keys()})
        if num_events == max_events: break

    out_file.close()
        
if __name__ == "__main__":
    skim()