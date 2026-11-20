import sys, os, ctypes
os.add_dll_directory(r"C:\Projects\kyojin\.venv-win\Lib\site-packages\_rocm_sdk_core\bin"); ctypes.CDLL(sys.argv[1])
import torch
G=2**30; pin=int(sys.argv[2]); mode=sys.argv[3]
torch.zeros(1,device='cuda')
hold=[]
try:
    for i in range(pin):
        if mode=="pinned": hold.append(torch.empty(G,dtype=torch.uint8,pin_memory=True)); hold[-1].fill_(1)
        else: hold.append(torch.ones(G,dtype=torch.uint8))
    f,t=torch.cuda.mem_get_info(); print(f"host {mode} {pin} GiB held; free {f/G:.1f}",flush=True)
except Exception as e: print("host alloc failed at",len(hold),str(e)[:80]); sys.exit()
xs=[]
try:
    while len(xs)<100:
        x=torch.empty(G,dtype=torch.uint8,device="cuda"); x.fill_(1); torch.cuda.synchronize(); xs.append(x)
except Exception as e: print(f"  device wall: failed after {len(xs)} GiB ({type(e).__name__})")
else: print("  reached 100 GiB")
