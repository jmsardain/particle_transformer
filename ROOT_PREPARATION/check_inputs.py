import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

file_to_check = "/mnt/iusers01/fatpou01/phy01/j17668am/scratch/particle_transformer/ROOT_PREPARATION/train.root:tree"
max_jets = int(1e6)

with uproot.open(file_to_check) as f:
    labels = f["label_sig"].array()
    
    jet_pt = f["jet_pt"].array()
    weights = f["weight"].array()
    
    sig = jet_pt[labels==1][:max_jets]
    sig_w = weights[labels==1][:max_jets]
    bkg = jet_pt[labels==0][:max_jets]
    bkg_w = weights[labels==0][:max_jets]
    
    print(np.sum(bkg_w))
    print(np.sum(sig_w))
    
    ax = plt.gca()
    
    #Plot UNWEIGHTED
    n, bins, patches = ax.hist(bkg, 100, label='QCD', linewidth=2, histtype='step', linestyle='--')
    ax.hist(sig, bins=bins, label=r"$Z' \rightarrow t\bar{t}$", linewidth=2, histtype='step')
    ax.set_xlabel("UNWEIGHTED JET_PT")

    ax.set_ylabel("Counts")
    ax.legend()
    ax.minorticks_off()
    plt.tight_layout()
    plt.savefig('UNWEIGHTED_JET_PT.png', dpi=500)
    plt.clf()
    
    ax = plt.gca()

    #Plot WEIGHTED
    n, bins, patches = ax.hist(bkg, weights=bkg_w, bins=100, label='QCD', linewidth=2, histtype='step', linestyle='--')
    ax.hist(sig, weights=sig_w, bins=bins, label=r"$Z' \rightarrow t\bar{t}$", linewidth=2, histtype='step')
    ax.set_xlabel("WEIGHTED JET_PT")

    ax.set_ylabel("Counts")
    ax.legend()
    ax.minorticks_off()
    plt.tight_layout()
    plt.savefig('WEIGHTED_JET_PT.png', dpi=500)
    plt.clf()

    #hep.atlas.text('Simulation Preliminary', ax=ax)
    #ax.text(0.05, 0.895, r"$\sqrt{s} = 13$ TeV, Pythia8", transform=ax.transAxes, ha='left', va='top')
    #ax.text(0.05, 0.838, r"anti-$k_t$, $R=1.0$ UFO SD jets", transform=ax.transAxes, ha='left', va='top')
