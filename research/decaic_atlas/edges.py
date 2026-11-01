import json, re, sys, collections
sys.path.insert(0,'.')
import group as G
LIST = {"ki": {"sabosid": {3:120,5:120,8:60,4:420}, "mebosid/gabosid": {3:360,4:180,5:120,6:60,8:60,10:60},
               "apbosid": {3:360,4:300,8:120}, "sigwebosid": {3:300,4:300,7:60}, "gigwebosid": {3:240,4:360,8:60},
               "tetbosid": {3:300,4:120,5:60,6:60}, "petbosid": {3:420,4:60,5:60,6:60}},
        "io": {"sosbid": {3:120,4:300,5:240}, "gosbid": {3:360,4:60,5:240,6:60}, "tetosbid": {3:240,4:240},
               "petosbid": {3:360,4:180}}}[G.SUB]
d = json.load(open(G.fname('snub_map')))
walls = collections.defaultdict(set)
for w in d['walls']:
    if w['wall_type'] not in w['sides']: walls[w['wall_type']].add(w['surface'][:40])
for t in d['types']:
    sg = t['signature']
    if sg.startswith('none'): continue
    nv = int(sg.split(':')[0])
    edges = collections.Counter()
    for n, k in re.findall(r'(\d+)x(\d+)', sg.split('edges')[1]): edges[int(k)] += int(n)
    cells = sum(int(a) for a in re.findall(r'(\d+)x\[', sg.split('|')[0]))
    match = [n for n, e in LIST.items() if e == dict(edges)]
    kind = 'region' if t['grid_regions'] else ('fold' if t['fold_points'] else 'wall')
    print(f"{t['type']:2} {kind:6} {nv}v {cells}c pts {sum(t['grid_regions']) or t['fold_points']:5} edges {dict(sorted(edges.items()))}  -> {match or '-'}  {sorted(walls.get(t['type'], []))[:2]}")
