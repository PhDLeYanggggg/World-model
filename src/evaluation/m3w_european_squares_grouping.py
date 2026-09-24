"""Outcome-blind locality groups; a grouping is not a split or independence proof."""
from itertools import combinations
import math
import unicodedata
from urllib.parse import parse_qs, urlparse


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def camera_identity(url):
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or '').casefold().removeprefix('www.')
    if host == 'youtu.be':
        video = parsed.path.strip('/').split('/')[0]
        return 'youtube:' + video if video else None
    if host in ('youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            video = parse_qs(parsed.query).get('v', [''])[0]
        elif parsed.path.startswith(('/embed/', '/live/', '/shorts/')):
            video = parsed.path.split('/')[2]
        else:
            video = ''
        return 'youtube:' + video if video else None
    return (host + parsed.path.rstrip('/') + ('?' + parsed.query if parsed.query else '')) if host else None


def geographic_distance_km(a, b):
    lat1, lon1, lat2, lon2 = a['latitude'], a['longitude'], b['latitude'], b['longitude']
    if any(v is None or not math.isfinite(v) for v in (lat1,lon1,lat2,lon2)):
        return None
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError('Invalid source geographic location')
    lat1,lon1,lat2,lon2=map(math.radians,(lat1,lon1,lat2,lon2))
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088*2*math.asin(math.sqrt(min(1.0,h)))


def locality_groups(sites, radius_km=5.0):
    if not math.isfinite(radius_km) or radius_km < 0:
        raise ValueError('Invalid source-group radius')
    sites=sorted(sites,key=lambda r:r['square_id'])
    ids=[r['square_id'] for r in sites]
    if len(set(ids))!=len(ids):
        raise ValueError('Repeated square identity')
    parent={sid:sid for sid in ids}
    def root(sid):
        while parent[sid]!=sid:
            sid=parent[sid]
        return sid
    edges=[]
    for a,b in combinations(sites,2):
        reasons=[]
        if normalized(a['city']) and normalized(a['city'])==normalized(b['city']) and normalized(a['country'])==normalized(b['country']):
            reasons.append('same_overview_city_country')
        c1,c2=camera_identity(a['camera_url']),camera_identity(b['camera_url'])
        if c1 is not None and c1==c2:
            reasons.append('same_publisher_camera_reference')
        distance=geographic_distance_km(a,b)
        if distance is not None and distance<=radius_km:
            reasons.append('geographic_proximity')
        if reasons:
            left,right=root(a['square_id']),root(b['square_id'])
            parent[max(left,right)]=min(left,right)
            edges.append(dict(square_ids=[a['square_id'],b['square_id']],reasons=reasons,
                              source_location_distance_km=distance))
    groups={}
    for sid in ids:
        groups.setdefault(root(sid),[]).append(sid)
    return dict(radius_km=radius_km,groups=list(groups.values()),edges=edges)
