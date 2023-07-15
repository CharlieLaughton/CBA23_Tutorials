#!/usr/bin/env python

from WElib import Walker, CrossflowPMEMDCudaStepper, Simple1DBinner, SplitMerger, Recycler, Checkpointer, FunctionProgressCoordinator
from crossflow.clients import Client
import mdtraj as mdt
import numpy as np
import yaml
import argparse

def pc_func(state, topology):
    '''
    Given an AMBER state, return the distance between the Na and Cl atoms
    
    '''
    t = mdt.load(state, top=topology)
    d = mdt.compute_distances(t, [[0, 1]])
    return d[0, 0]

def run(client, config):
    mdin = config['mdin']
    inpcrd = config['inpcrd']
    prmtop = config['prmtop']
    n_reps = config['n_reps']
    n_cycles = config['n_cycles']
    target_pc = config['target_pc']
    logfile = config['logfile']
    edges = config['edges']
    checkpointdir = config['checkpointdir']
    check_freq = config['check_freq']
    restart = config['restart']
    
    reftraj = mdt.load(inpcrd, top=prmtop)

    stepper = CrossflowPMEMDCudaStepper(client, mdin, prmtop)
    pc = FunctionProgressCoordinator(pc_func, reftraj.topology)
    binner = Simple1DBinner(edges)
    recycler = Recycler(inpcrd, target_pc, retrograde=True)
    splitmerger = SplitMerger(n_reps)

    checkpointer = Checkpointer(checkpointdir, mode='rw')
    if restart:
        walkers = checkpointer.load()
    else:
        weight = 1.0 / n_reps
        walkers = [Walker(inpcrd, weight) for i in range(n_reps)]
        #checkpointer.save(walkers)
    walkers = pc.run(walkers)
    with open(logfile, 'w') as f:
        pcs = np.array([w.pc for w in walkers])
        f.write('{}\n'.format(pcs))
        for i in range(n_cycles):
            walkers = stepper.run(walkers)
            walkers = pc.run(walkers)
            pcs = np.array([w.pc for w in walkers])
            f.write('{}\n'.format(pcs))
            front = pcs.min()
            walkers = binner.run(walkers)
            walkers, flux1 = recycler1.run(walkers)
            if flux1 > 0.0:
                walkers = pc.run(walkers)
            walkers, flux2 = recycler2.run(walkers)
            if flux2  > 0.0:
                walkers = pc.run(walkers)
            walkers = binner.run(walkers)
            walkers = splitmerger.run(walkers)
            f.write('{} {} {} {}\n'.format(i, front, flux2, len(walkers)))
            f.flush()
            if i % check_freq == 0:
                checkpointer.save(walkers)

    checkpointer.save(walkers)

parser = argparse.ArgumentParser()
parser.add_argument('configfile', help='Configuration file (YAML format)')
args = parser.parse_args()
with open(args.configfile) as f:
    config = yaml.safe_load(f)
    client = Client(scheduler_file=config['scheduler_file'])
    run(client, config)
