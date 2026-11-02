"""Плавные графики: интерполяция меняет отображение, а не расчётную сетку."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

CMAP = 'inferno'
BACKGROUND = '#f8fafc'
INK = '#334155'


def setup_axis(ax, data):
    size = data['p']['L'] * 100
    ax.set(xlim=(0, size), ylim=(0, size), zlim=(0, size),
           xlabel='x, см', ylabel='y, см', zlabel='z, см')
    ax.set_box_aspect((1, 1, 1), zoom=.92)
    ax.set_proj_type('ortho')
    ax.set_facecolor(BACKGROUND)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.label.set(color=INK, fontsize=10)
        axis.set_pane_color((.96, .97, .99, 0.28))
        axis.line.set_color('#cbd5e1')
        axis._axinfo['grid'].update(color=(.76, .80, .85, .35), linewidth=.6)
    ax.tick_params(colors='#64748b', labelsize=9, pad=1)
    pts = np.array([[0,0,0], [size,0,0], [size,size,0], [0,size,0],
                    [0,0,size], [size,0,size], [size,size,size], [0,size,size]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot(*pts[[i,j]].T, color='#94a3b8', lw=.7, alpha=.6)
    outlines = {'x-':[0,3,7,4], 'x+':[1,2,6,5], 'y-':[0,1,5,4],
                'y+':[3,2,6,7], 'z+':[4,5,6,7]}
    for face in data['faces']:
        ix = outlines[face] + outlines[face][:1]
        ax.plot(*pts[ix].T, color='#0891b2', lw=1.8, alpha=.9)
    lo = (data['p']['L'] - data['p']['a']) * 50
    hi = (data['p']['L'] + data['p']['a']) * 50
    heater = [(lo,lo,0),(hi,lo,0),(hi,hi,0),(lo,hi,0)]
    ax.add_collection3d(Poly3DCollection(
        [heater], facecolor='#fb7185', edgecolor='#be123c', linewidth=.9, alpha=.28))


def normalization(data):
    lo, hi = float(data['T'].min()), float(data['T'].max())
    return Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.)


def _isotherm_meshes(data):
    """Кэшируем геометрию: поворот и повторный показ не требуют marching cubes."""
    if '_isotherm_meshes' in data:
        return data['_isotherm_meshes']
    T = data['T']
    norm = normalization(data)
    meshes = []
    if np.ptp(T) > 0:
        n = T.shape[0]
        count = max(n, 46)
        grid = np.linspace(0, n - 1, count)
        coords = np.array(np.meshgrid(grid, grid, grid, indexing='ij'))
        field = map_coordinates(T, coords, order=1, mode='nearest', prefilter=False)
        step = (n - 1) * data['h'] * 100 / (count - 1)
        for fraction in (.18, .39, .60, .81):
            level = norm.vmin + fraction * (norm.vmax - norm.vmin)
            vertices, faces, _, _ = marching_cubes(field, level=level, spacing=(step,)*3)
            vertices += data['h'] * 50
            meshes.append((level, vertices[faces]))
    data['_isotherm_meshes'] = meshes
    return meshes


def draw_isotherms(ax, data):
    norm = normalization(data)
    cmap = colormaps[CMAP]
    handles = []
    for level, triangles in _isotherm_meshes(data):
        color = cmap(norm(level))
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolor=color, edgecolor='none', linewidth=0,
            antialiased=True, alpha=.58, rasterized=True))
        handles.append(Patch(facecolor=color, edgecolor='none', label=f'{level:.1f} °C'))
    setup_axis(ax, data)
    if handles:
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, .99),
                  frameon=False, fontsize=9, labelcolor=INK, handlelength=1.1,
                  handleheight=1.1, labelspacing=.7)
    return norm


def draw_slices(ax, data, indices):
    """Дробные индексы дают непрерывное перемещение плоскостей."""
    T = data['T']
    n = T.shape[0]
    count = 56  # Ограничиваем сложность 3D независимо от расчётной сетки.
    size = data['p']['L'] * 100
    coord = np.linspace(0, size, count)
    A, B = np.meshgrid(coord, coord, indexing='ij')
    # В крайних половинах ячеек продолжаем ближайшее значение центра.
    I, J = np.meshgrid(np.clip(coord / (data['h'] * 100) - .5, 0, n - 1),
                       np.clip(coord / (data['h'] * 100) - .5, 0, n - 1), indexing='ij')
    norm = normalization(data)
    cmap = colormaps[CMAP]
    artists = []
    for axis, index in enumerate(indices):
        index = float(np.clip(index, 0, n - 1))
        fixed = np.full_like(A, (index + .5) * data['h'] * 100)
        sample = np.full_like(I, index)
        coords = [fixed,A,B] if axis == 0 else [A,fixed,B] if axis == 1 else [A,B,fixed]
        sample_coords = [sample,I,J] if axis == 0 else [I,sample,J] if axis == 1 else [I,J,sample]
        values = map_coordinates(T, np.array(sample_coords), order=1,
                                 mode='nearest', prefilter=False)
        # Цвет задаём в центрах полигонов, чтобы не сдвигать карту на полшага.
        centers = .25 * (values[:-1,:-1] + values[1:,:-1]
                         + values[:-1,1:] + values[1:,1:])
        artist = ax.plot_surface(*coords, facecolors=cmap(norm(centers)),
                                 rstride=1, cstride=1, linewidth=0,
                                 antialiased=False, shade=False, alpha=1,
                                 rasterized=True)
        artists.append(artist)
    return norm, artists
