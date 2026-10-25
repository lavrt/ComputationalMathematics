"""Проверка параметров и расчёт без зависимости от интерфейса."""
import math
import time
from thermal_model import ThermalModel

DEFAULTS = dict(L=.05, a=.03, P=200., lam=200., T_water=20., alpha=2000., N=16)
FACE_NAMES = {'x-':'−X','x+':'+X','y-':'−Y','y+':'+Y','z+':'+Z (верх)'}


def validate(raw, faces):
    limits = {'L':(.01,.2), 'a':(.001,.19), 'P':(.1,2000), 'lam':(1,1000),
              'T_water':(-20,100), 'alpha':(100,100000), 'N':(8,40)}
    p = {}
    for key, (lo,hi) in limits.items():
        try: value=float(str(raw.get(key,DEFAULTS[key])).replace(',','.'))
        except (ValueError,TypeError): raise ValueError(f'{key}: требуется число') from None
        if not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'{key}: допустимо от {lo} до {hi}')
        p[key]=value
    if p['N']!=int(p['N']): raise ValueError('N должно быть целым')
    p['N']=int(p['N'])
    if p['a']>=p['L']: raise ValueError('Сторона процессора должна быть меньше стороны куба')
    if not faces or any(f not in FACE_NAMES for f in faces) or len(set(faces))!=len(faces):
        raise ValueError('Выберите хотя бы одну из пяти охлаждаемых сторон')
    return p


def solve(p, faces):
    started=time.perf_counter()
    model=ThermalModel(**{k:p[k] for k in ['L','a','P','lam','T_water']})
    h,d,b,power,parity=model.make_grid(p['N'],faces,p['alpha'])
    u,history=model.sor(d,b,parity,omega=1.97,eps=1e-8)
    tmax,mean,flows=model.metrics(u,h,power,faces,p['alpha'])
    if abs(sum(flows.values())-p['P'])/p['P']>1e-5:
        raise RuntimeError('Не выполнен тепловой баланс')
    return dict(p=p,faces=list(faces),T=p['T_water']+u,h=h,tmax=tmax,mean=mean,
                flows=flows,iterations=len(history),seconds=time.perf_counter()-started)
