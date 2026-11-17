"""Operational fix: initialize CUDA before resetting allocator counters.

Does not alter the frozen codec/scientific source or completed records.
"""
import argparse
import os
from pathlib import Path
import runpy

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--gpu',type=int,default=0)
args,_=p.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
torch.cuda.set_device(0)
torch.cuda.init()
torch.cuda.set_per_process_memory_fraction(.05,0)
runpy.run_path(str(Path(__file__).with_name('run.py')),run_name='__main__')
