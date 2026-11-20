import sys, time, torch
sys.path.insert(0, "C:/Projects/kyojin")
from exllamav3.loader.safetensors import SafetensorsCollection
stc = SafetensorsCollection("C:/models/flash-next-4p05")
keys = sorted(stc.tensor_file_map.keys())
sel = [k for k in keys if "hc" in k.split(".")[0] or k.startswith("hc")][:20] + [k for k in keys if ".layers.0." in k][:int(sys.argv[1])]
print("keys", len(sel), "first", sel[:3], flush=True)
dev = torch.device("cuda:0"); t0 = time.time()
for i, k in enumerate(sel):
    t = stc.get_tensor(k, dev, allow_bf16=False)
    if i % 50 == 0: print("got", i, k, str(t.dtype), tuple(t.shape), f"t={time.time()-t0:.1f}s", flush=True)
torch.cuda.synchronize(); print("DONE", len(sel))
