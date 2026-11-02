"""Настольное приложение. Запуск: python app.py"""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from calculation import DEFAULTS, FACE_NAMES, validate, solve
from graphics import BACKGROUND, CMAP, setup_axis, draw_isotherms, draw_slices


class Application:
    def __init__(self, root):
        self.root=root
        root.title('Охлаждение процессора')
        root.geometry('1250x850')
        root.minsize(1000,720)
        root.configure(bg=BACKGROUND)
        style=ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background=BACKGROUND, foreground='#334155')
        style.configure('TFrame', background=BACKGROUND)
        style.configure('TLabel', background=BACKGROUND)
        style.configure('TNotebook', background=BACKGROUND, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(18, 10))
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')],
                  foreground=[('selected', '#0891b2')])
        style.configure('TButton', padding=(12, 10), background='#0891b2', foreground='white')
        style.map('TButton', background=[('active', '#0e7490'), ('disabled', '#cbd5e1')])
        style.configure('TEntry', padding=5, fieldbackground='white')
        self.data=None
        self.busy=False
        self.closed=False
        self.pending=None
        self.setting_sliders=False
        self.messages=queue.Queue()
        self.selected={'z+'}
        self.entries={}
        self.buttons={}
        root.columnconfigure(1,weight=1)
        root.rowconfigure(0,weight=1)
        left=ttk.Frame(root,padding=18)
        left.grid(row=0,column=0,sticky='ns')
        ttk.Label(left,text='Охлаждаемые грани',font=('',11,'bold')).grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,10))
        for i,(face,name) in enumerate(FACE_NAMES.items()):
            b=tk.Button(left,text=name,width=10,command=lambda f=face:self.toggle_face(f),
                        relief='flat',font=('',10,'bold'),padx=5,pady=7)
            b.grid(row=1+i//2,column=i%2,padx=3,pady=3,sticky='ew')
            self.buttons[face]=b
        self.update_buttons()
        ttk.Label(left,text='Параметры расчёта',font=('',11,'bold')).grid(row=4,column=0,columnspan=2,sticky='w',pady=(18,10))
        labels=[('L','Сторона куба, м'),('a','Сторона процессора, м'),('P','Мощность, Вт'),
                ('lam','Теплопроводность, Вт/(м·К)'),('T_water','Температура воды, °C'),
                ('alpha','Теплоотдача α, Вт/(м²·К)'),('N','Ячеек вдоль ребра, N')]
        for i,(key,label) in enumerate(labels):
            row=5+2*i
            ttk.Label(left,text=label).grid(row=row,column=0,columnspan=2,sticky='w',pady=(7,3))
            entry=ttk.Entry(left,width=27)
            entry.insert(0,str(DEFAULTS[key]))
            entry.grid(row=row+1,column=0,columnspan=2,sticky='ew')
            self.entries[key]=entry
        self.submit=ttk.Button(left,text='Рассчитать',command=self.calculate)
        self.submit.grid(row=19,column=0,columnspan=2,sticky='ew',pady=(18,0))
        right=ttk.Frame(root,padding=(0,10,10,10))
        right.grid(row=0,column=1,sticky='nsew')
        right.columnconfigure(0,weight=1)
        right.rowconfigure(0,weight=1)
        self.tabs=ttk.Notebook(right)
        self.tabs.grid(row=0,column=0,sticky='nsew')
        self.figures=[]
        self.canvases=[]
        self.plot_axes=[None, None]
        self.slice_artists=[]
        for title in ['Изотермические поверхности','Температурные сечения']:
            tab=ttk.Frame(self.tabs)
            self.tabs.add(tab,text=title)
            fig=Figure(figsize=(8,6),dpi=100,facecolor=BACKGROUND)
            ax=fig.add_subplot(projection='3d')
            ax.set_axis_off()
            canvas=FigureCanvasTkAgg(fig,master=tab)
            canvas.get_tk_widget().pack(fill='both',expand=True)
            canvas.mpl_connect('scroll_event',lambda event,c=canvas:self.zoom(event,c))
            self.figures.append(fig)
            self.canvases.append(canvas)
        self.slider_frame=ttk.Frame(right,padding=(15,8))
        self.scales=[]
        self.labels=[]
        for axis in range(3):
            label=ttk.Label(self.slider_frame,text='xyz'[axis])
            label.grid(row=axis,column=0,sticky='w',padx=(0,12))
            scale=tk.Scale(self.slider_frame,from_=0,to=15,orient='horizontal',showvalue=False,
                           state='disabled',resolution=.02, bd=0, highlightthickness=0,
                           bg=BACKGROUND, troughcolor='#e2e8f0', activebackground='#0891b2',
                           sliderrelief='flat', sliderlength=22, width=8,
                           command=lambda value:self.schedule_slices())
            scale.grid(row=axis,column=1,sticky='ew')
            self.scales.append(scale)
            self.labels.append(label)
        self.slider_frame.columnconfigure(1,weight=1)
        self.tabs.bind('<<NotebookTabChanged>>',self.tab_changed)
        root.bind('<Return>',lambda event:self.calculate())
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_id=root.after(100,self.poll)

    def update_buttons(self):
        for face,button in self.buttons.items():
            active=face in self.selected
            button.configure(bg='#0891b2' if active else '#e2e8f0',
                             fg='white' if active else '#334155',
                             activebackground='#22b8cf' if active else '#cbd5e1')

    def toggle_face(self,face):
        if self.busy:return
        if face in self.selected:self.selected.remove(face)
        else:self.selected.add(face)
        self.update_buttons()

    def calculate(self):
        if self.busy:return
        try:
            faces=[f for f in FACE_NAMES if f in self.selected]
            p=validate({key:entry.get() for key,entry in self.entries.items()},faces)
        except ValueError as exc:
            messagebox.showerror('Параметры расчёта',str(exc),parent=self.root)
            return
        self.busy=True
        self.submit.configure(text='Расчёт…',state='disabled')
        for widget in [*self.entries.values(),*self.buttons.values()]:widget.configure(state='disabled')

        def worker():
            try:self.messages.put(('done',solve(p,faces)))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            kind,value=self.messages.get_nowait()
        except queue.Empty:
            self.poll_id=self.root.after(100,self.poll)
            return
        try:
            if kind=='error':
                messagebox.showerror('Ошибка расчёта',value,parent=self.root)
            else:
                self.data=value
                self.setting_sliders=True
                for scale in self.scales:
                    scale.configure(to=value['p']['N']-1,state='normal')
                    scale.set(value['p']['N']//2)
                self.setting_sliders=False
                self.plot_axes=[None, None]
                self.slice_artists=[]
                self.render(0)
                self.render(1)
                self.root.title(f"Охлаждение процессора — Tmax контакта {value['tmax']:.2f} °C")
        except Exception as exc:
            messagebox.showerror('Ошибка графика',str(exc),parent=self.root)
        finally:
            self.setting_sliders=False
            self.busy=False
            self.submit.configure(text='Рассчитать',state='normal')
            for widget in [*self.entries.values(),*self.buttons.values()]:widget.configure(state='normal')
        self.poll_id=self.root.after(100,self.poll)

    def render(self,index):
        if self.closed or self.data is None:return
        fig=self.figures[index]
        ax=self.plot_axes[index]
        indices=[float(s.get()) for s in self.scales]
        if ax is None:
            # При новом расчёте сохраняем камеру, но возвращаем границы нового куба.
            old=next((a for a in fig.axes if a.name=='3d'),None)
            elev,azim=(old.elev,old.azim) if old else (25,-55)
            fig.clear()
            ax=fig.add_subplot(projection='3d')
            ax.view_init(elev=elev,azim=azim)
            self.plot_axes[index]=ax
            if index==0:
                norm=draw_isotherms(ax,self.data)
            else:
                setup_axis(ax,self.data)
                norm,self.slice_artists=draw_slices(ax,self.data,indices)
            bar=fig.colorbar(ScalarMappable(norm=norm,cmap=CMAP),ax=ax,
                            shrink=.62,pad=.065,fraction=.035,aspect=28)
            bar.set_label('Температура, °C',color='#334155',fontsize=10,labelpad=12)
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors='#64748b',labelsize=9,width=0)
            fig.subplots_adjust(left=.01,right=.90,bottom=.04,top=.99)
        elif index==1:
            # Сохраняем оси, шкалу, поворот и масштаб; заменяем только три плоскости.
            limits=(ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d())
            for artist in self.slice_artists:artist.remove()
            _,self.slice_artists=draw_slices(ax,self.data,indices)
            ax.set_xlim3d(limits[0]); ax.set_ylim3d(limits[1]); ax.set_zlim3d(limits[2])
        if index==1:
            for axis,i in enumerate(indices):
                cm=(i+.5)*self.data['h']*100
                self.labels[axis].configure(text=f"{'xyz'[axis]} = {cm:.2f} см")
        self.canvases[index].draw_idle()

    def schedule_slices(self):
        if self.setting_sliders or self.data is None or self.closed:return
        if self.pending is None:
            self.pending=self.root.after(45,self.redraw_slices)

    def redraw_slices(self):
        self.pending=None
        self.render(1)

    def tab_changed(self,event=None):
        if self.tabs.index(self.tabs.select())==1:
            self.slider_frame.grid(row=1,column=0,sticky='ew')
        else:self.slider_frame.grid_remove()

    def zoom(self,event,canvas):
        ax=event.inaxes
        if ax is None or ax.name!='3d':return
        factor=.85 if event.button=='up' else 1/.85
        for getter,setter in [(ax.get_xlim3d,ax.set_xlim3d),(ax.get_ylim3d,ax.set_ylim3d),(ax.get_zlim3d,ax.set_zlim3d)]:
            lo,hi=getter(); center=(lo+hi)/2;half=(hi-lo)*factor/2
            setter(center-half,center+half)
        canvas.draw_idle()

    def close(self):
        self.closed=True
        self.root.after_cancel(self.poll_id)
        if self.pending is not None:self.root.after_cancel(self.pending)
        self.root.destroy()


if __name__=='__main__':
    root=tk.Tk()
    Application(root)
    root.mainloop()
