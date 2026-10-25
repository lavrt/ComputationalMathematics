"""Настольное приложение. Запуск: python app.py"""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from calculation import DEFAULTS, FACE_NAMES, validate, solve
from graphics import draw_isotherms, draw_slices


class Application:
    def __init__(self, root):
        self.root=root
        root.title('Охлаждение процессора')
        root.geometry('1250x850')
        root.minsize(1000,720)
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
        for title in ['Изотермические поверхности','Температурные сечения']:
            tab=ttk.Frame(self.tabs)
            self.tabs.add(tab,text=title)
            fig=Figure(figsize=(8,6),dpi=100,facecolor='white')
            ax=fig.add_subplot(projection='3d')
            ax.set(xlabel='x, см',ylabel='y, см',zlabel='z, см')
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
                           state='disabled',command=lambda value:self.schedule_slices())
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
        # Рабочий поток не обращается к Tk и Matplotlib.
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
        old=next((a for a in fig.axes if a.name=='3d'),None)
        elev,azim=(old.elev,old.azim) if old else (25,-60)
        fig.clear()
        ax=fig.add_subplot(projection='3d')
        ax.view_init(elev=elev,azim=azim)
        if index==0:norm=draw_isotherms(ax,self.data)
        else:
            indices=[int(s.get()) for s in self.scales]
            norm=draw_slices(ax,self.data,indices)
            for axis,i in enumerate(indices):
                cm=(i+.5)*self.data['h']*100
                self.labels[axis].configure(text=f"{'xyz'[axis]} = {cm:.2f} см")
        fig.colorbar(ScalarMappable(norm=norm,cmap='inferno'),ax=ax,shrink=.65,pad=.09,label='T, °C')
        fig.subplots_adjust(left=0,right=.94,bottom=.08,top=.98)
        self.canvases[index].draw_idle()

    def schedule_slices(self):
        if self.setting_sliders or self.data is None or self.closed:return
        if self.pending is not None:self.root.after_cancel(self.pending)
        # Перерисовываем только сечения, не извлекая изоповерхности повторно.
        self.pending=self.root.after(150,self.redraw_slices)

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
