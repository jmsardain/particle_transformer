# Particle Transformer

This code is forked from [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).

## Setup:
It is able to run PFN, ParticleNet, Particle Transformer and also PELICAN.

To setup:
```bash
git clone git@github.com:AlexEdmundMay/particle_transformer.git
cd particle_transformer
git clone git@github.com:AlexEdmundMay/weaver-core.git
```

Create a conda environment (I have used conda-forge):

```bash
conda create --name weaver --file weaver-core/requirements.txt
```

## configs:

The training file is [train_JetClass.sh](https://github.com/AlexEdmundMay/particle_transformer/blob/main/train_JetClass.sh). This file submits a job using the command 'weaver'.
Note: I have experienced some trouble training on multigpu, so this may require some tinkering if NGPU is set > 1.
```
--data-train: The file(s) used for training (root files work best: see section on ROOT_PREPARATION to see how to get this data from TDD h5 files or Dumper/Flattener root files)
--data-test: File(s) used for testing.
--data-fraction: What fraction of the training set should you use to train. e.g. if full train set is 90 million jets, a value of 0.1 would train on 9 million each epoch (selected randomly)
--train-val-split: What fraction of the training set do you want to reserve for validation?
--data-config: Location of config to load data (more details to come).
--network-config: location of config file for each network (this is where you define which loss function to use, and can tune hyperparameters)
--model-prefix: Where to save the model checkpoints
--num-epochs: number of training epochs
--optimizer: Which optimiser to use during training
--log: File location to output the training log file
--predict-output: After training, inference is run in the test set: This is the filename of the test set.
--tensorboard: Save location of tensorboard file
--predict: If this flag is included, then weaver will only run inference on the test set, and will ignore the train set (you must set model-prefix to a particular model checkpoint)
```
ADDITIONAL options can be found in [weaver-core/weaver/train.py](https://github.com/AlexEdmundMay/weaver-core/blob/main/weaver/train.py)

The data-configs tell weaver how to read the train and test files. The config [JetClass_kin](https://github.com/AlexEdmundMay/particle_transformer/blob/main/data/JetClass/JetClass_kin.yaml) has been editted to work with the outputs of ROOT_PREPARATION.

There are multiple sections to these configs:
```
selection: performs selections on samples (I perform selections in ROOT_PREPARATION instead)
new_variables: Here you can define new variables using simple equations (and numpy functions)
inputs: This is where you define the networks points, features, vectors and mask
labels: Labels used for training (If you add additional labels, make sure you add them here)
observers: When running inference on the test set, this adds these values to the outputted root file
weights: Weights to use when sampling training set
```
Note: by default weaver samples the training set using it's weight. If instead you want to weight the loss function directly, then add an observer called "weight" in data config, set --load_observers_during_training True in train_JetClass.sh and in your network config loss, set (reduction = 'none'). Hence, do not use the key 'weight' in the observers section if you want to continue using sampling.

## To Run:

I have a submission script called [ParT_job.sh](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ParT_job.sh) which is used to submit to the CSF3 University of Manchester cluster.
This defines which model to use (ParticleNet='PN', ParticleTransformer='ParT',PFN='PFN', PELICAN='PELICAN'), the number of epochs to train for and the data location.
It is submitted using:
```bash
qsub ParT_job.sh
```
This will need to be adjusted if you want to submit to a different cluster.

## ROOT_PREPARATION:

The ROOT_PREPARATION folder lets you create weaver-compatible root files from either TDD hdf5 files or root files from the Dumper/Flattener.

### from TDD h5:
NOTE: this is not yet very versatile: there are several hardcoded integer values to define locations for the jet-level variables.

To run this, set the relevant fields in the [run_skim_tdd.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/run_skim_tdd.py), then:
```bash
python run_skim_tdd.py
python match_weights.py
```

If you need to add fields, this can be done in [hdf5_to_root.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/hdf5_to_root.py) in the skim(...) function.
To edit the selection cuts, see [prep_functions.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/prep_functions.py)

run_skim_tdd.py: outputs a number of output files for signal and bkg depending on 'num_outputs'. These are skimmed and have selection cuts applied.
match_weights.py: takes these output files and combines signal and background, shuffles them, producing single train and test files.

Note: If the hdf5 files are updated, make sure you check the function define_jet_level_quantities(...), and the definition of the variable 'mcEventWeight' in skim(...), since these have hard coded integer indeces.
You will also have to change the branch names used in [hdf5_to_root.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/hdf5_to_root.py), if the TDD h5 branch names are updated.


### from Dumper/Flattener Root Files:
This is only slightly more versatile than the h5:
In [run_skim.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/run_skim.py): define all the branches that are needed in the processing in 'branches_to_use_in_preprocess' and which branches you want to keep in the output branches in 'branches_to_keep'. If the hdf5 branch names change, update this file. (If "fjet_clus_pt" or "fjet_clus_E" change, also update in [process_samples.py](https://github.com/AlexEdmundMay/particle_transformer/blob/main/ROOT_PREPARATION/process_samples.py))

```bash
python run_skim.py
python match_weights.py
```
