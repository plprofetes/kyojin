import torch
xs=[]; G=2**30
f0,t=torch.cuda.mem_get_info(); print("start free",round(f0/G,2),"total",round(t/G,2),flush=True)
try:
    for i in range(1,100):
        x=torch.empty(G,dtype=torch.uint8,device="cuda"); x.fill_(i%250); torch.cuda.synchronize(); xs.append(x)
        if i%4==0 or i>=34:
            f,_=torch.cuda.mem_get_info(); print(f"alloc {i} GiB (synced+touched) free {f/G:.2f} consumed {(t-f)/G:.2f}",flush=True)
except Exception as e:
    print("FAIL at",len(xs),"GiB:",type(e).__name__,str(e)[:120],flush=True)
