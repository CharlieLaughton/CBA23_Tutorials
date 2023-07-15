#!/usr/bin/env python
# coding: utf-8

from crossflow.tasks import SubprocessTask
from crossflow.filehandling import FileHandler
from crossflow.clients import Client
import mdtraj as mdt
from pathlib import Path
import yaml
import sys
import os

with open(sys.argv[1]) as f:
    config = yaml.load(f, yaml.SafeLoader)

runmd = SubprocessTask(
    "pmemd.cuda -AllowSmallBox -i md.in -o md.log -c start.ncrst -r final.ncrst -p system.prmtop -x md.nc"
)
runmd.set_inputs(["md.in", "start.ncrst", "system.prmtop", "disang.dat"])
runmd.set_outputs(["md.log", "final.ncrst", "md.nc"])

fh = FileHandler()
mdin = fh.load(config["mdin"])
prmtop = fh.load(config["prmtop"])
template = Path(config["disang_template"]).read_text()
start_crdfile = config["startcrds"]
metadatafile = config["metadata"]
r_min = config["r_min"]
r_max = config["r_max"]
r_fac = config["r_fac"]
r_k = config["r_k"]

disang = Path("disang.dat")


def create_disang(template, r):
    """
    Create the content for a disang file from the temp[ate

    ]"""
    params = {"r1": max(1.0, r - 6.0), "r2": r, "r4": r + 6.0, "r_k": r_k}
    return template.format(**params)


runmd.set_constant("md.in", mdin)
runmd.set_constant("system.prmtop", prmtop)

if __name__ == "__main__":
    client = Client(n_workers=1)
    startcrds = fh.load(start_crdfile)
    r = r_min
    cycle = 0
    if os.path.exists(metadatafile):
        print('*** Appending new windows to existsing metadata file ***')
        with open(metadatafile) as f1:
            data = f1.readlines()
            cycle = len(data)

    with open(metadatafile, "a") as f1:
        while r < r_max:
            cycle += 1
            print(f"\n*** Starting umbrella sampling window {cycle} ***")
            print(
                f"  Umbrella restraint parameters:\n    r_0 = {r:6.3f}\n    r_k = {r_k:6.3f}\n"
            )
            disang.write_text(create_disang(template, r))
            log, finalcrds, trajfile = client.submit(runmd, startcrds, disang)
            try:
                t = mdt.load(trajfile.result(), top=prmtop)
            except:
                print("  Whoops! MD run failed - stopping.")
                raise
            tfilename = f"cycle_{cycle:03d}.nc"
            rfilename = f"cycle_{cycle:03d}.ncrst"
            dfilename = f"cycle_{cycle:03d}.dist"
            print("  MD simulation complete,")
            print(f"  Writing trajectory file {tfilename}")
            trajfile.result().save(tfilename)
            print(f"  Writing restart file {rfilename}")
            finalcrds.result().save(rfilename)
            d = mdt.compute_distances(t, [[0, 1]]) * 10.0  # to Angstroms
            print(f"  Writing distance file {dfilename}")
            with open(dfilename, "w") as f2:
                for frame, dd in enumerate(d):
                    if frame > 9:  # Ignore first 10 values - equilibration
                        f2.write("{:4.1f} {}\n".format(float(frame), dd[0]))
            f1.write(f"{dfilename} {r} {r_k*2}\n")  # Double r_k - see WHAM manual
            dm = d.mean()
            ds = d.std()
            print(f"  Mean distance this cycle: {dm:6.3f} SD = {ds:6.3f}\n")
            r = r + ds * r_fac
            startcrds = finalcrds
        print("R_max reached, ending simulations.")
