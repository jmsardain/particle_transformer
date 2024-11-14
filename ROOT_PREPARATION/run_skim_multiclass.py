import hdf5_to_root as h5_tr
import numpy as np
config_WZ={
    "out_file_dir" : "/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/data_out_qcd_for_jad/",
    "pickle_dict_file":"/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/metadata/jet_count_dict_MULTICLASS.pickle",

    "max_constits" : 80,
    "max_jets" : int(45e6),
    "num_outputs" : 50,
    "num_test" : 5, #num_train = num_outputs-num_test
    
    "R22TruthLabelValues":[2,3,4,5],    
    "signal":True, 
    "labels":{"label_QCD":0,"label_WZ":1,"label_top":0,"label_higgs":0},
    "branches_to_keep" : ["fjet_clus_E","fjet_clus_pt","fjet_clus_phi",
                        "fjet_clus_eta","fjet_clus_deltaphi","fjet_clus_deltaeta",
                        "fjet_pt","jet_pt","jet_energy","R10TruthLabel_R22v1","fjet_testing_weight_pt","weight"],
    "out_file_tag":"WZ",
    "metadata_file":"/eos/user/a/almay/particle_transformer/ROOT_PREPARATION/metadata/metadata.json",
    "in_file" :["/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.801859.*/*.h5"],
}
config_WZ["branches_to_keep"]= np.append(config_WZ["branches_to_keep"],list(config_WZ["labels"].keys()))

config_H={
    "out_file_dir" : config_WZ["out_file_dir"],
    "pickle_dict_file":config_WZ["pickle_dict_file"],
    "branches_to_keep" : config_WZ["branches_to_keep"],
    "max_constits" : config_WZ["max_constits"],
    "max_jets" : config_WZ["max_jets"],
    "num_outputs" : config_WZ["num_outputs"],
    "num_test" : config_WZ["num_test"], #num_train = num_outputs-num_test
    "metadata_file":config_WZ["metadata_file"],

    "R22TruthLabelValues":[11],
    "signal":True, 
    "labels":{"label_QCD":0,"label_WZ":0,"label_top":0,"label_higgs":1},
    "out_file_tag":"H",
    "in_file" :["/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.426351*/user.rles.*.output.h5"],
}
config_top={
    "out_file_dir" : config_WZ["out_file_dir"],
    "pickle_dict_file":config_WZ["pickle_dict_file"],
    "branches_to_keep" : config_WZ["branches_to_keep"],
    "max_constits" : config_WZ["max_constits"],
    "max_jets" : config_WZ["max_jets"],
    "num_outputs" : config_WZ["num_outputs"],
    "num_test" : config_WZ["num_test"], #num_train = num_outputs-num_test
    "metadata_file":config_WZ["metadata_file"],

    "R22TruthLabelValues":[1],
    "signal":True, 
    "labels":{"label_QCD":0,"label_WZ":0,"label_top":1,"label_higgs":0},
    "out_file_tag":"top",
    "in_file" :["/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.801661*/user.rles.*.output.h5"],
}

config_bkg={
    "out_file_dir" : config_WZ["out_file_dir"],
    "pickle_dict_file":config_WZ["pickle_dict_file"],
    "branches_to_keep" : config_WZ["branches_to_keep"],
    "max_constits" : config_WZ["max_constits"],
    "max_jets" : config_WZ["max_jets"],
    "num_outputs" : config_WZ["num_outputs"],
    "num_test" : config_WZ["num_test"], #num_train = num_outputs-num_test
    "metadata_file":config_WZ["metadata_file"],
    
    "R22TruthLabelValues":[10],
    "signal":True, 
    "labels":{"label_QCD":1,"label_WZ":0,"label_top":0,"label_higgs":0},
    "out_file_tag":"qcd",
    "in_file" :["/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.36470[3-9].e7142_s3681_r13144_p5548.tdd.FatJetsFlow_jetm2.24_2_31.24-04-17_JetTagging-1-g079895f_output.h5/user.rles.*.output.h5","/eos/atlas/atlascerngroupdisk/perf-jets/TAGGING/TDDh5/LargeR_June12_2024/user.rles.36471[0-2].e7142_s3681_r13144_p5548.tdd.FatJetsFlow_jetm2.24_2_31.24-04-17_JetTagging-1-g079895f_output.h5/user.rles.*.output.h5"],
}

h5_tr.run(config_WZ)
h5_tr.run(config_H)
h5_tr.run(config_top)
h5_tr.run(config_bkg)

