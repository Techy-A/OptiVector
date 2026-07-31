import vtracer, resvg_py, numpy as np, io, os, itertools
from PIL import Image
os.makedirs('.tmp/sw', exist_ok=True)

def score(inp, **kw):
    out=f".tmp/sw/t.svg"
    vtracer.convert_image_to_svg_py(inp, out, **kw)
    sz=os.path.getsize(out)/1024
    txt=open(out).read(); npaths=txt.count('<path')
    raw=resvg_py.svg_to_bytes(svg_path=out, width=512, height=512, background='#ffffff')
    a=np.asarray(Image.open(io.BytesIO(bytes(raw))).convert('RGB').resize((512,512)),np.float32)/255
    b=np.asarray(Image.open(inp).convert('RGB').resize((512,512)),np.float32)/255
    return round(float(np.abs(a-b).mean()),4), round(sz,0), npaths

print(f"{'cp':>3}{'spk':>5}{'ld':>4} {'mode':>8} | {'L1':>7}{'KB':>7}{'paths':>7}")
for cp,spk,ld,mode in itertools.product([6,8],[2,8],[16,48],['spline','polygon']):
    try:
        l1,kb,p=score('phoenix.png', colormode='color', color_precision=cp, filter_speckle=spk, layer_difference=ld, mode=mode, path_precision=2)
        print(f"{cp:>3}{spk:>5}{ld:>4} {mode:>8} | {l1:>7}{kb:>7.0f}{p:>7}")
    except Exception as e:
        print(cp,spk,ld,mode,'ERR',str(e)[:40])
