"""Two-decision opening rule with a source-held continuation target."""
import numpy as np


def sample(options,atoms):
    x = np.stack([z[3] for z in options])
    selected = np.stack([atoms[z[1],z[2]] for z in options])
    ref,easy,harm,_,_,utility = selected.T
    supported = (easy>0.) & (harm<=.02*ref)
    y = np.where(supported,(utility>0.).astype(float),-1.)
    return x,y


def ridge(samples, width):
    """Each source has equal weight, each recording equal weight inside source."""
    grams, rhs = [], []
    for source in samples:
        useful = [(x,y) for x,y in source if len(x)]
        if not useful:
            continue
        grams.append(np.mean([x.T@x/len(x) for x,_ in useful],axis=0))
        rhs.append(np.mean([x.T@y/len(x) for x,y in useful],axis=0))
    if not grams:
        return np.zeros(width)
    penalty = np.eye(width)*.02
    penalty[0,0] = 0.
    return np.linalg.solve(np.mean(grams,axis=0)+penalty,np.mean(rhs,axis=0))


def fit(by_source, width):
    stop = ridge(list(by_source.values()),width)
    future = []
    for name, rows in by_source.items():
        other = ridge([r for s,r in by_source.items() if s!=name],width)
        values = []
        for x,y in rows:
            target = np.zeros(len(y))
            # If there is no next eligible query, waiting yields abstention.
            target[:-1] = np.where(x[1:]@other>0.,y[1:],0.)
            values.append((x,target))
        future.append(values)
    return stop,ridge(future,width)


def plan(options, stop, wait):
    """Return thresholdable decisions; no future feature affects the first decision."""
    if not options:
        return [], []
    _,i,j,x = options[0]
    s = float(np.clip(x@stop,-1.,1.))
    w = float(np.clip(x@wait,-1.,1.))
    current = [(s,i,j,None)]
    if s>=w:
        return current,current
    if len(options)<2:
        return current,[]
    _,i,j,next_x = options[1]
    return current,[(min(w,float(np.clip(next_x@stop,-1.,1.))),i,j,None)]
