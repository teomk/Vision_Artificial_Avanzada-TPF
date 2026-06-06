import os
import shutil

def parse_filename(fname):
    name = fname.replace(".tif", "")
    parts = name.split("_")
    roi    = parts[0]
    season = parts[1]
    patch  = parts[-1]
    num    = parts[-2]
    return roi, season, num, patch

def build_index(folder):
    index = {}
    for f in os.listdir(folder):
        try:
            roi, season, num, patch = parse_filename(f)
            key = (roi, season, num, patch)
            index[key] = f
        except:
            pass
    return index

s1_idx     = build_index("data/south_america_s1")
s2_idx     = build_index("data/south_america_s2")
cloudy_idx = build_index("data/south_america_s2_cloudy")

triples = set(s1_idx.keys()) & set(s2_idx.keys()) & set(cloudy_idx.keys())
solo_s1 = set(s1_idx.keys()) - set(s2_idx.keys()) - set(cloudy_idx.keys())

print(f"Triples completos: {len(triples)}")
print(f"S1 sin contraparte (a mover): {len(solo_s1)}")

# Crear carpeta destino
dest = "data/s1_incompletas"
os.makedirs(dest, exist_ok=True)

# Mover
for key in solo_s1:
    fname = s1_idx[key]
    src_path  = os.path.join("data/south_america_s1", fname)
    dest_path = os.path.join(dest, fname)
    
    # Dry run — comentar para ejecutar de verdad
    print(f"  Movería: {src_path} → {dest_path}")
    continue
    # shutil.move(src_path, dest_path)

print(f"Movidos {len(solo_s1)} archivos a '{dest}'")