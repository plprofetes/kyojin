import sys, os, ctypes
os.add_dll_directory(r"C:\Projects\kyojin\.venv-win\Lib\site-packages\_rocm_sdk_core\bin"); ctypes.CDLL(sys.argv[1])
import torch
G=2**30; c=float(sys.argv[2]); M=2**20; cm=int(c*1024); xs=[]; tot=0
f0,t=torch.cuda.mem_get_info(); print("total",round(t/G,2),"free",round(f0/G,2),flush=True)
try:
    while tot < 100*1024:
        x=torch.empty(cm*M,dtype=torch.uint8,device="cuda"); x.fill_(1); torch.cuda.synchronize(); xs.append(x); tot+=cm
        if (tot//1024)%10==0 and tot%1024==0: print(f"  {tot/1024:.0f} GiB ok, free {torch.cuda.mem_get_info()[0]/G:.1f}",flush=True)
except Exception as e:
    print(f"chunk {cm} MiB: failure after {tot/1024:.1f} GiB: {type(e).__name__}: {str(e)[:80]}")
else: print(f"chunk {cm} MiB: reached {tot/1024:.1f} GiB, no failure")
