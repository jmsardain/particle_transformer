import process_samples as ps

config_sig={
    "tree" : ":FlatSubstructureJetTree",
    "out_file_dir" : "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/data_out/",
    "branches_to_keep" : ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi",
                        "fjet_clus_eta","fjet_clus_deltaphi","fjet_clus_deltaeta",
                        "label_QCD","label_sig","jet_pt","jet_energy","weight"],
    "branches_to_use_in_preprocess" : ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi","fjet_clus_eta", 'fjet_truthJet_eta','fjet_truthJet_pt',
                                    'fjet_numConstituents','fjet_m','fjet_truth_dRmatched_particle_flavor','fjet_truth_dRmatched_particle_dR',
                                    'fjet_truthJet_dRmatched_particle_dR_top_W_matched','fjet_ungroomed_truthJet_m','fjet_ungroomed_truthJet_Split23',
                                    'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount','fjet_ungroomed_truthJet_pt'],
    "max_constits" : 80,
    "max_jets" : int(250000),
    "num_outputs" : 50,
    "num_test" : 5, #num_train = num_outputs-num_test
        
    "signal":True, 
    "out_file_tag":"sig",
    "in_file" :"/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/ntuples/user.almay.32*.tree.root",
}

config_bkg={
    "tree" : ":FlatSubstructureJetTree",
    "out_file_dir" : "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/data_out/",
    "branches_to_keep" : config_sig["branches_to_keep"],
    "branches_to_use_in_preprocess" : config_sig["branches_to_use_in_preprocess"],
    "max_constits" : config_sig["max_constits"],
    "max_jets" : config_sig["max_jets"],
    "num_outputs" : config_sig["num_outputs"],
    "num_test" : config_sig["num_test"], #num_train = num_outputs-num_test
    
    "signal":False, 
    "out_file_tag":"bkg",
    "in_file" :"/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/ntuples/user.almay.33*.tree.root",
}

ps.run(config_sig)
ps.run(config_bkg)