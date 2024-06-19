import hdf5_to_root as h5_tr

config_sig={
    "tree" : ":FlatSubstructureJetTree",
    "out_file_dir" : "/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/data_out_bkg_only/",
    "pickle_dict_file":"/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/metadata/jet_count_dict.pickle",
    "branches_to_keep" : ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi",
                        "fjet_clus_eta","fjet_clus_deltaphi","fjet_clus_deltaeta",
                        "label_QCD","label_sig","fjet_pt","jet_pt","jet_energy","fjet_testing_weight_pt","weight"],
    "branches_to_use_in_preprocess" : ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi","fjet_clus_eta", "fjet_testing_weight_pt",'fjet_truthJet_eta','fjet_truthJet_pt',
                                    'fjet_numConstituents',"fjet_pt",'fjet_m','fjet_truth_dRmatched_particle_flavor','fjet_truth_dRmatched_particle_dR',
                                    'fjet_truthJet_dRmatched_particle_dR_top_W_matched','fjet_ungroomed_truthJet_m','fjet_ungroomed_truthJet_Split23',
                                    'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount','fjet_ungroomed_truthJet_pt'],
    "max_constits" : 80,
    "max_jets" : int(1e6),
    "num_outputs" : 1,
    "num_test" : 1, #num_train = num_outputs-num_test
        
    "signal":True, 
    "out_file_tag":"sig",
    "metadata_file":"/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/metadata/metadata.json",
    "in_file" :"/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/ntuples/user.almay.32*.tree.root",
}

config_bkg={
    "tree" : ":FlatSubstructureJetTree",
    "out_file_dir" : config_sig["out_file_dir"],
    "pickle_dict_file":config_sig["pickle_dict_file"],
    "branches_to_keep" : config_sig["branches_to_keep"],
    "branches_to_use_in_preprocess" : config_sig["branches_to_use_in_preprocess"],
    "max_constits" : config_sig["max_constits"],
    "max_jets" : config_sig["max_jets"],
    "num_outputs" : config_sig["num_outputs"],
    "num_test" : config_sig["num_test"], #num_train = num_outputs-num_test
    "metadata_file":config_sig["metadata_file"],

    "signal":False, 
    "out_file_tag":"bkg",
    "in_file" :"/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.36471[0-2].e7142_s3681_r13144_p5548.tdd.FatJetsFlow_jetm2.24_2_31.24-04-17_JetTagging-1-g079895f_output.h5/user.rles.*._0000*.output.h5",
}

#h5_tr.run(config_sig)
h5_tr.run(config_bkg)
