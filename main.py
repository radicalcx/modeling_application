import tkinter as tk
import typing
from functools import partial
import numpy as np
import sympy as sp
import numpy.typing
from numpy.random import exponential, rand
import pandas as pd
from scipy.integrate import odeint
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


class VerticalScrolledFrame(tk.LabelFrame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame.
    * Construct and pack/place/grid normally.
    * This frame only allows vertical scrolling.
    """

    def __init__(self, parent, *args, **kw):
        tk.LabelFrame.__init__(self, parent, *args, **kw)

        # Create a canvas object and a vertical scrollbar for scrolling it.
        vscrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=tk.FALSE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0,
                           yscrollcommand=vscrollbar.set, height=200)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        vscrollbar.config(command=canvas.yview)

        # Reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # Create a frame inside the canvas which will be scrolled with it.
        self.interior = interior = tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=tk.NW)

        # Track changes to the canvas and frame width and sync them,
        # also updating the scrollbar.
        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        # canvas.bind('<Configure>', _configure_canvas)


class Interaction:
    def __init__(self, inp_arg, out_arg):
        self.inp = np.array(inp_arg)
        self.out = np.array(out_arg)
        self.prob_intervals = [1, ]
        self.prob = [1, ]

    def add_probabilities(self, prob_arg: numpy.typing.NDArray):
        self.prob = prob_arg
        self.prob_intervals = np.array([prob_arg[0]])
        for el in prob_arg[1:]:
            self.prob_intervals = np.append(self.prob_intervals, self.prob_intervals[-1] + el)

    def choose_out(self):
        if len(self.prob_intervals) == 1:
            return self.out[0]
        else:
            coin = rand()
            num = (coin < self.prob_intervals).argmax()
            return self.out[num]


def create_trajectory(inter: typing.List[Interaction], init_val, lam, time, m):
    trajectory = np.array([init_val])
    time_array = np.zeros(1)
    while time_array[-1] < time:
        temp_lam = [inter[i].inp * trajectory[-1] for i in range(m)]
        tau = pd.Series(
            [exponential(1 / (np.prod(temp_lam[i][temp_lam[i] > 0]) * lam[i]))] for i in range(m)).sort_values()
        for num, t in tau.items():
            if np.greater_equal(trajectory[-1], inter[num].inp).all():
                trajectory = np.append(trajectory, [trajectory[-1] - inter[num].inp + inter[num].choose_out()], axis=0)
                time_array = np.append(time_array, time_array[-1] + t)
                break
        else:
            trajectory = np.append(trajectory, [trajectory[-1]], axis=0)
            time_array = np.append(time_array, time_array[-1] + 1)

    trajectory[-1] = trajectory[-2]
    time_array[-1] = time
    return trajectory, time_array


def calculate_math_expectation(inter: typing.List[Interaction], init_val, lam, time, n, m):
    s = [sp.Symbol('s' + str(i)) for i in range(n)]
    arg_for_subs = [(s[i], 1) for i in range(n)]
    components = [sum(
        [sp.prod(
            [s[j] ** inter[i].out[k][j] for j in range(n)]) * inter[i].prob[k] for k in range(len(inter[i].out))]) -
                  sp.prod(
                      [s[j] ** inter[i].inp[j] for j in range(n)]) for i in range(m)]
    derivatives = [inter[i].inp.copy() for i in range(m)]

    diff_components = [[sp.diff(components[i], s[j]) for j in range(n)] for i in range(m)]
    diff_derivatives = [[derivatives[j].copy() for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            diff_derivatives[i][j][i] += 1

    components = [components[i].subs(arg_for_subs) for i in range(m)]
    diff_components = [[diff_components[i][j].subs(arg_for_subs) for j in range(n)] for i in range(m)]

    def system(x, t, lamb):
        res = [sum([
            lamb[i] * (
                    diff_components[i][j] *
                    np.prod([x[k] ** derivatives[i][k] for k in range(n)])
                    +
                    components[i] *
                    np.prod([x[k] ** diff_derivatives[j][i][k] for k in range(n)])
            )
            for i in range(m)]) for j in range(n)]
        return res

    return odeint(system, init_val, np.linspace(0, time, time), args=(lam,))


def main():
    frm_main = tk.Tk()
    frm_main.title('modeling_application')
    frm_main.grid_columnconfigure(0, weight=1)
    frm_main.grid_columnconfigure(1, weight=8)

    frm_sizes = tk.LabelFrame(frm_main, text='Размерности системы', padx=15, pady=10)
    frm_sizes.grid(row=0, column=0, padx=15, pady=1, sticky=tk.NSEW)

    lbl_n = tk.Label(frm_sizes, text='Количество типов элементов')
    lbl_m = tk.Label(frm_sizes, text='Количество комплексов взаимодействия')

    ent_n = tk.Entry(frm_sizes, width=2)
    ent_m = tk.Entry(frm_sizes, width=2)

    lbl_n.grid(row=0, column=0, padx=15, pady=5)
    ent_n.grid(row=0, column=1, padx=15, pady=5)

    lbl_m.grid(row=1, column=0, padx=15, pady=5)
    ent_m.grid(row=1, column=1, padx=15, pady=5)

    def init_frm_complexes(n_str, m_str):
        try:
            n = int(n_str)
            m = int(m_str)
        except Exception as ex:
            return

        interactions_with_prob = {}

        frm_complexes_base = tk.LabelFrame(frm_main, padx=15, pady=10, text='Комплексы взаимодействия')
        frm_complexes = VerticalScrolledFrame(frm_complexes_base, padx=15, pady=10)
        frm_complex_rows = [tk.Frame(frm_complexes.interior, padx=3, pady=0) for i in range(m)]
        frm_out = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]
        frm_out_text = [[tk.Frame(frm_out[i], padx=0, pady=0), ] for i in range(m)]
        frm_inp_text = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]
        frm_inp_btn = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]

        frm_complexes_base.grid(row=1, column=0, padx=15, pady=1, sticky=tk.NSEW)
        frm_complexes.pack()
        for i in range(m):
            frm_complex_rows[i].pack(padx=15, pady=5)
            frm_inp_text[i].grid(row=0, column=0, padx=0, sticky=tk.NSEW)
            frm_inp_btn[i].grid(row=1, column=0, padx=0, sticky=tk.NSEW)
            frm_out[i].grid(row=0, column=1, padx=0, sticky=tk.NSEW)
            frm_out_text[i][0].pack(side=tk.TOP, padx=0, pady=0, expand=True)

        # nonlocal ent_inp_values, ent_out_values
        ent_inp_values = [[tk.Entry(frm_inp_text[i], width=1) for j in range(n)] for i in range(m)]
        ent_out_values = [[[tk.Entry(frm_out_text[i][0], width=1) for j in range(n)], ] for i in range(m)]

        px = 0
        py = 3
        for i in range(m):
            for j in range(n - 1):
                ent_inp_values[i][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_inp_values[i][j].insert(0, '0')
                tk.Label(frm_inp_text[i], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_inp_values[i][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_inp_values[i][n - 1].insert(0, '0')
            tk.Label(frm_inp_text[i], text='T' + str(n) + ' \N{RIGHTWARDS BLACK ARROW}').pack(side=tk.LEFT, padx=px,
                                                                                              pady=py)

        for i in range(m):
            for j in range(n - 1):
                ent_out_values[i][0][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_out_values[i][0][j].insert(0, '0')
                tk.Label(frm_out_text[i][0], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[i][0][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[i][0][n - 1].insert(0, '0')
            tk.Label(frm_out_text[i][0], text='T' + str(n)).pack(side=tk.LEFT, padx=px, pady=py)

        # nonlocal interactions_with_prob

        def add_complex(idx):
            frm_out_text[idx].append(tk.Frame(frm_out[idx], padx=15, pady=5))
            frm_out_text[idx][-1].pack()

            ent_out_values[idx].append([tk.Entry(frm_out_text[idx][-1], width=1) for j in range(n)])
            for j in range(n - 1):
                ent_out_values[idx][-1][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_out_values[idx][-1][j].insert(0, '0')
                tk.Label(frm_out_text[idx][-1], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[idx][-1][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[idx][-1][n - 1].insert(0, '0')
            tk.Label(frm_out_text[idx][-1], text='T' + str(n)).pack(side=tk.LEFT, padx=px, pady=py)

            interactions_with_prob[idx] = interactions_with_prob.setdefault(idx, 1) + 1

        btn_add_complex = [tk.Button(frm_inp_btn[i], text='Добавить', command=partial(add_complex, i)) for i in
                           range(m)]
        for i in range(m):
            btn_add_complex[i].pack(side=tk.TOP)

        def init_interactions():

            interactions = [(Interaction(
                inp_arg=np.array([int(ent_inp_values[i][j].get()) for j in range(n)]),
                out_arg=np.array(
                    [[int(ent_out_values[i][k][j].get()) for j in range(n)] for k in range(len(ent_out_values[i]))])
            )) for i in range(m)]


            frm_params_base = tk.LabelFrame(frm_main, text='Параметры системы', pady=0)
            frm_params_base.grid(row=2, column=0, sticky=tk.NSEW)

            frm_params = VerticalScrolledFrame(frm_params_base)
            frm_params.pack(side=tk.TOP, expand=True, pady=1)

            frm_init_values = tk.Frame(frm_params.interior, padx=15, pady=10)
            frm_lam = tk.Frame(frm_params.interior, padx=15, pady=10)
            frm_prob = {key: tk.Frame(frm_params.interior, padx=15, pady=10) for key in interactions_with_prob}
            frm_time = tk.Frame(frm_params.interior, padx=15, pady=10)
            frm_count = tk.Frame(frm_params.interior, padx=15, pady=10)

            frm_init_values.pack()
            frm_lam.pack()
            for el in frm_prob.values():
                el.pack()
            frm_time.pack()
            frm_count.pack()

            ent_init_values = [tk.Entry(frm_init_values, width=8) for i in range(n)]
            ent_lam = [tk.Entry(frm_lam, width=8) for i in range(m)]
            ent_prob = {key: [tk.Entry(frm_prob[key], width=3) for i in range(count)]
                        for key, count in interactions_with_prob.items()}

            for i, el in enumerate(ent_init_values):
                tk.Label(frm_init_values, text='T' + str(i + 1) + ' ').pack(side=tk.LEFT)
                el.pack(side=tk.LEFT)

            for i, el in enumerate(ent_lam):
                tk.Label(frm_lam, text='lam' + str(i + 1) + '=').pack(side=tk.LEFT)
                el.pack(side=tk.LEFT)

            for key, ent in ent_prob.items():
                for i, el in enumerate(ent):
                    tk.Label(frm_prob[key], text='p' + str(key + 1) + str(i + 1) + '=').pack(side=tk.LEFT)
                    el.pack(side=tk.LEFT)

            ent_time = tk.Entry(frm_time, width=4)
            ent_count = tk.Entry(frm_count, width=4)

            tk.Label(frm_time, text='T=').pack(side=tk.LEFT)
            ent_time.pack(side=tk.LEFT)

            tk.Label(frm_count, text='N=').pack(side=tk.LEFT)
            ent_count.pack(side=tk.LEFT)

            def init_params():
                # nonlocal time, probabilities, lam, init_values, count

                try:
                    init_values = np.array([int(ent_init_values[i].get()) for i in range(n)])
                    lam = np.array([float(ent_lam[i].get()) for i in range(m)])
                    time = int(ent_time.get())
                    probabilities = {num: np.array(list(map(lambda x: float(x.get()), values))) for num, values in
                                     ent_prob.items()}
                    count = int(ent_count.get())
                except Exception as ex:
                    print(ex)

                if any([False if abs(sum(val) - 1) < 1e-8 else True for val in probabilities.values()]):
                    return
                else:
                    for num, arr in probabilities.items():
                        interactions[num].add_probabilities(arr)

                fig = Figure(figsize=(5, 5),
                             dpi=100)
                tr, t_a = create_trajectory(interactions, init_values, lam, time, m)

                plot1 = fig.add_subplot()
                plot1.plot(t_a, tr[:, 0])
                canvas = FigureCanvasTkAgg(fig,
                                           master=frm_main)
                canvas.draw()
                canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, sticky=tk.NSEW,)

                toolbar = NavigationToolbar2Tk(canvas, frm_main)
                toolbar.update()

                canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, sticky=tk.NSEW)
                # print(
                #     calculate_math_expectation(inter=interactions, init_val=init_values, lam=lam, time=time, n=n, m=m))

            tk.Button(frm_params_base, text='Рассчитать', command=init_params).pack(side=tk.TOP, pady=1)

        frm_btn_apply_complexes = tk.Frame(frm_complexes_base, padx=15, pady=10)
        frm_btn_apply_complexes.pack()
        btn_apply_complexes = tk.Button(frm_btn_apply_complexes, text='Применить', command=init_interactions)
        btn_apply_complexes.pack()

    btn_apply_sizes = tk.Button(frm_sizes, text='Применить',
                                command=lambda: init_frm_complexes(ent_n.get(), ent_m.get()))

    btn_apply_sizes.grid(row=2, column=1, columnspan=2)

    frm_main.mainloop()


if __name__ == '__main__':
    main()
