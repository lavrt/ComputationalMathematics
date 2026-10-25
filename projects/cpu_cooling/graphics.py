"""Графики Matplotlib. Можно проверить без окон с backend Agg."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def setup_axis(ax, data):
    s=data['p']['L']*100
    ax.set(xlim=(0,s),ylim=(0,s),zlim=(0,s),xlabel='x, см',ylabel='y, см',zlabel='z, см')
    ax.set_box_aspect((1,1,1))
    pts=np.array([[0,0,0],[s,0,0],[s,s,0],[0,s,0],
                  [0,0,s],[s,0,s],[s,s,s],[0,s,s]])
    for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot(*pts[[i,j]].T,color='#64748b',lw=.7)
    poly={'x-':[0,3,7,4],'x+':[1,2,6,5],'y-':[0,1,5,4],
          'y+':[3,2,6,7],'z+':[4,5,6,7]}
    for face in data['faces']:
        ix=poly[face]+[poly[face][0]]
        ax.plot(*pts[ix].T,color='#0891b2',lw=2)
    lo,hi=(data['p']['L']-data['p']['a'])*50,(data['p']['L']+data['p']['a'])*50
    heater=[(lo,lo,0),(hi,lo,0),(hi,hi,0),(lo,hi,0)]
    ax.add_collection3d(Poly3DCollection([heater],facecolor='#ef4444',alpha=.8))


def normalization(data):
    return Normalize(vmin=float(data['T'].min()),vmax=float(data['T'].max()))


def draw_isotherms(ax, data):
    T=data['T']; norm=normalization(data); cmap=colormaps['inferno']
    levels=np.linspace(norm.vmin+.18*(norm.vmax-norm.vmin),norm.vmax-.18*(norm.vmax-norm.vmin),4)
    handles=[]
    for level in levels:
        vertices,faces,_,_=marching_cubes(T,level=level,spacing=(data['h']*100,)*3)
        vertices+=data['h']*50
        color=cmap(norm(level))
        ax.add_collection3d(Poly3DCollection(vertices[faces],facecolor=color,edgecolor='none',alpha=.4))
        handles.append(Patch(facecolor=color,label=f'{level:.1f} °C',alpha=.6))
    setup_axis(ax,data)
    ax.legend(handles=handles,loc='upper left',fontsize=8)
    return norm


def draw_slices(ax, data, indices):
    T=data['T']; n=T.shape[0]; coord=(np.arange(n)+.5)*data['h']*100
    A,B=np.meshgrid(coord,coord,indexing='ij')
    norm=normalization(data); cmap=colormaps['inferno']
    for axis,index in enumerate(indices):
        fixed=np.full((n,n),coord[index])
        coords=[fixed,A,B] if axis==0 else [A,fixed,B] if axis==1 else [A,B,fixed]
        values=np.take(T,index,axis=axis)
        ax.plot_surface(*coords,facecolors=cmap(norm(values)),rstride=1,cstride=1,
                        linewidth=0,antialiased=False,shade=False,alpha=.93)
    setup_axis(ax,data)
    return norm
