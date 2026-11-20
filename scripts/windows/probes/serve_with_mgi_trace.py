import sys, runpy, torch
orig = torch.cuda.mem_get_info
def mgi(dev=None):
    f,t = orig(dev) if dev is not None else orig()
    fr = sys._getframe(1)
    if 'model_ls' in fr.f_code.co_filename:
        print(f"[mgi] free {f/2**30:.2f} total {t/2**30:.2f} alloc {torch.cuda.memory_allocated()/2**30:.2f} reserved {torch.cuda.memory_reserved()/2**30:.2f}", flush=True)
    return f,t
torch.cuda.mem_get_info = mgi
sys.argv = ["serve.py"] + sys.argv[1:]
runpy.run_path("tools/qwen/serve.py", run_name="__main__")
