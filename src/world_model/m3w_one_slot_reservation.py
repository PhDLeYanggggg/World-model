"""Past-only wait rule with a single earned intervention slot."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as ledger


def allocate(groups, value, a, e, legal, mint):
    out = np.full(len(value), -1, np.int16)
    balance = np.zeros(2)
    price = None
    reservations = released = 0
    for g in groups:
        k = int((mint[g]>=0).sum())
        u, ac, ec, ok = (x[g] for x in (value,a,e,legal))
        current = ledger.query_choice(u,ac,ec,ok,balance,count=k)
        if current is None:
            raise ValueError('Original minted actions must fit the predicted local budget')
        total = ledger.totals(current,u,ac,ec)
        if price is None and k==2:
            first = ledger.query_choice(u,ac,ec,ok,balance,count=1)
            if first is not None:
                stats = ledger.totals(first,u,ac,ec)
                marginal = total[0]-stats[0]
                if stats[0]>0. and marginal>0.:
                    current,total,price = first,stats,marginal
                    reservations += 1
        elif price is not None and k<min(2,len(g)-1):
            extra = ledger.query_choice(u,ac,ec,ok,balance,count=k+1)
            if extra is not None:
                stats = ledger.totals(extra,u,ac,ec)
                if stats[0]-total[0]>price:
                    current,total,price = extra,stats,None
                    released += 1
        out[g] = current
        balance -= total[1:]
    return out, dict(held=reservations,released=released,unused=int(price is not None))
