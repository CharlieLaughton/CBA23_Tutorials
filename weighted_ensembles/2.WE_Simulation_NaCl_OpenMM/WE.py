#!/usr/bin/env python

from WElib import Walker, FunctionStepper, StaticBinner, SplitMerger, Recycler, Checkpointer, FunctionProgressCoordinator
from crossflow.clients import Client
from crossflow.tasks import SubprocessTask
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

md = SubprocessTask('pmemd.cuda -i md.in -c start.ncrst -p system.prmtop -r final.ncrst -AllowSmallBox')
md.set_inputs(['md.in', 'start.ncrst', 'system.prmtop'])
md.set_outputs(['final.ncrst'])

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
    md.set_constant('system.prmtop', prmtop)
    md.set_constant('md.in', mdin)

    def runmd(start, client, md):
        final = client.submit(md, start)
        return final.result()

    stepper = FunctionStepper(runmd, client, md)
    pc = FunctionProgressCoordinator(pc_func, reftraj.topology)
    binner = StaticBinner(edges)
    recycler = Recycler(target_pc, retrograde=True)
    splitmerger = SplitMerger(n_reps)

    #checkpointer = Checkpointer(checkpointdir, mode='rw')
    if restart:
        walkers = checkpointer.load()
    else:
        weight = 1.0 / n_reps
        walkers = [Walker(inpcrd, weight) for i in range(n_reps)]
        #checkpointer.save(walkers)
    new_walkers = pc.run(walkers)
    with open(logfile, 'w') as f:
        print(' cycle    n_walkers   left-most bin  right-most bin   flux')
        for i in range(n_cycles):
            new_walkers = stepper.run(new_walkers)
            new_walkers = pc.run(new_walkers)
            new_walkers = binner.run(new_walkers)
            new_walkers = recycler.run(new_walkers)
            if recycler.flux > 0.0:
                new_walkers = pc.run(new_walkers)
            new_walkers = binner.run(new_walkers)
            new_walkers = splitmerger.run(new_walkers)
            occupied_bins = list(binner.bin_weights.keys())
            print(f' {i:3d} {len(new_walkers):10d} {min(occupied_bins):12d} {max(occupied_bins):14d} {recycler.flux:20.8f}')
            f.write(f' {i:3d} {len(new_walkers):10d} {min(occupied_bins):12d} {max(occupied_bins):14d} {recycler.flux:20.8f}\n')


parser = argparse.ArgumentParser()
parser.add_argument('configfile', help='Configuration file (YAML format)')
args = parser.parse_args()
with open(args.configfile) as f:
    config = yaml.safe_load(f)
    if __name__ == '__main__':
        client = Client(n_workers=1)
        run(client, config)
