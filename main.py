import customtkinter as tk
# import tkinter as tk
import typing
from functools import partial
import numpy as np
import sympy as sp
import numpy.typing
from numpy.random import exponential, rand
import pandas as pd
from scipy.stats import norm
from scipy.integrate import odeint
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

tk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
tk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


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


class Trajectory:
    def __init__(self, track_arg, time_arg):
        self.track = track_arg
        self.time = time_arg


class MyCanvas(FigureCanvasTkAgg):
    def __init__(self, figure, master, mean, samples,
                 trajectories_draw: typing.List[Trajectory], type_of_plot, idx: int, time, M, bins):
        super().__init__(figure, master)
        plot = self.figure.add_subplot()
        if type_of_plot == 'diagram':
            plot.hist(samples[:, idx],
                      bins=bins,
                      range=(samples[:, idx].min(), samples[:, idx].max()),
                      density='True',
                      color='blue')
            grid = np.arange(-3, 3, 0.1)
            plot.plot(grid, norm.pdf(grid, 0, 1), color='black', linewidth=3.0)
        elif type_of_plot == 'trajectory':
            plot.plot(np.linspace(0, time, time), mean[:, idx], color='red', linewidth=5, zorder=M + 1, linestyle='--')
            for tr in trajectories_draw:
                plot.plot(tr.time, tr.track[:, idx], linewidth=2)
        else:
            return
        self.draw()

    def change_plot(self, mean, samples,
                    trajectories_draw: typing.List[Trajectory], type_of_plot, idx: int, time, M, bins):
        self.figure.clf()
        plot = self.figure.add_subplot()

        if type_of_plot == 'diagram':
            plot.hist(samples[:, idx],
                      bins=bins,
                      range=(samples[:, idx].min(), samples[:, idx].max()),
                      density='True',
                      color='blue')
            grid = np.arange(-3, 3, 0.1)
            plot.plot(grid, norm.pdf(grid, 0, 1), color='black', linewidth=3.0)
        elif type_of_plot == 'trajectory':
            plot.plot(np.linspace(0, time, time), mean[:, idx], color='red', linewidth=5, zorder=M + 1, linestyle='--')
            for tr in trajectories_draw:
                plot.plot(tr.time, tr.track[:, idx], linewidth=2)
        else:
            return
        self.draw()


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

    return Trajectory(trajectory, time_array)


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


def modeling(inter: typing.List[Interaction], init_val, lam, time, n, m, N, M):
    samples = np.empty((N, n))
    trajectories_draw = [create_trajectory(inter, init_val, lam, time, m) for i in range(M)]
    for i in range(M):
        samples[i] = trajectories_draw[i].track[-1]
    for i in range(M, N):
        samples[i] = create_trajectory(inter, init_val, lam, time, m).track[-1]

    mean = calculate_math_expectation(inter, init_val, lam, time, n, m)
    std = samples.std(axis=0)

    for i in range(n):
        samples[:, i] = (samples[:, i] - mean[-1][i]) / (std[i] + 1e-10)

    return mean, samples, trajectories_draw


def calculate_chi2(sample: numpy.typing.NDArray):
    pass


def main():
    frm_main = tk.CTk()
    frm_main.title('modeling_application')
    frm_main.grid_columnconfigure(0, weight=1)
    frm_main.grid_columnconfigure(1, weight=8)
    frm_main.grid_rowconfigure((0, 1, 2), weight=1)

    frm_sizes = tk.CTkFrame(frm_main)
    frm_sizes.grid(row=0, column=0, padx=15, pady=1, sticky=tk.NSEW)

    lbl_n = tk.CTkLabel(frm_sizes, text='Количество типов элементов')
    lbl_m = tk.CTkLabel(frm_sizes, text='Количество комплексов взаимодействия')

    ent_n = tk.CTkEntry(frm_sizes, width=2)
    ent_m = tk.CTkEntry(frm_sizes, width=2)

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

        frm_complexes_base = tk.CTkFrame(frm_main)
        # frm_complexes = VerticalScrolledFrame(frm_complexes_base, padx=15, pady=10)
        frm_complexes = tk.CTkScrollableFrame(frm_complexes_base, width=500)
        frm_complex_rows = [tk.CTkFrame(frm_complexes) for i in range(m)]
        frm_out = [tk.CTkFrame(frm_complex_rows[i]) for i in range(m)]
        frm_out_text = [[tk.CTkFrame(frm_out[i]), ] for i in range(m)]
        frm_inp_text = [tk.CTkFrame(frm_complex_rows[i]) for i in range(m)]
        frm_inp_btn = [tk.CTkFrame(frm_complex_rows[i]) for i in range(m)]

        frm_complexes_base.grid(row=1, column=0, padx=15, pady=1, sticky=tk.NSEW)
        frm_complexes.pack()
        for i in range(m):
            frm_complex_rows[i].pack(padx=15, pady=5)
            frm_inp_text[i].grid(row=0, column=0, padx=0, sticky=tk.NSEW)
            frm_inp_btn[i].grid(row=1, column=0, padx=0, sticky=tk.NSEW)
            frm_out[i].grid(row=0, column=1, padx=0, sticky=tk.NSEW)
            frm_out_text[i][0].pack(side=tk.TOP, padx=0, pady=0, expand=True)

        # nonlocal ent_inp_values, ent_out_values
        ent_inp_values = [[tk.CTkEntry(frm_inp_text[i], width=1) for j in range(n)] for i in range(m)]
        ent_out_values = [[[tk.CTkEntry(frm_out_text[i][0], width=1) for j in range(n)], ] for i in range(m)]
        px = 0
        py = 3
        for i in range(m):
            for j in range(n - 1):
                ent_inp_values[i][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_inp_values[i][j].insert(0, '0')
                tk.CTkLabel(frm_inp_text[i], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_inp_values[i][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_inp_values[i][n - 1].insert(0, '0')
            tk.CTkLabel(frm_inp_text[i], text='T' + str(n) + ' \N{RIGHTWARDS BLACK ARROW}').pack(side=tk.LEFT, padx=px,
                                                                                                 pady=py)

        for i in range(m):
            for j in range(n - 1):
                ent_out_values[i][0][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_out_values[i][0][j].insert(0, '0')
                tk.CTkLabel(frm_out_text[i][0], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[i][0][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[i][0][n - 1].insert(0, '0')
            tk.CTkLabel(frm_out_text[i][0], text='T' + str(n)).pack(side=tk.LEFT, padx=px, pady=py)

        # nonlocal interactions_with_prob

        def add_complex(idx):
            frm_out_text[idx].append(tk.CTkFrame(frm_out[idx]))
            frm_out_text[idx][-1].pack()

            ent_out_values[idx].append([tk.CTkEntry(frm_out_text[idx][-1], width=1) for j in range(n)])
            for j in range(n - 1):
                ent_out_values[idx][-1][j].pack(side=tk.LEFT, padx=px, pady=py)
                ent_out_values[idx][-1][j].insert(0, '0')
                tk.CTkLabel(frm_out_text[idx][-1], text='T' + str(j + 1) + '+').pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[idx][-1][n - 1].pack(side=tk.LEFT, padx=px, pady=py)
            ent_out_values[idx][-1][n - 1].insert(0, '0')
            tk.CTkLabel(frm_out_text[idx][-1], text='T' + str(n)).pack(side=tk.LEFT, padx=px, pady=py)

            interactions_with_prob[idx] = interactions_with_prob.setdefault(idx, 1) + 1

        btn_add_complex = [tk.CTkButton(frm_inp_btn[i], text='Добавить', command=partial(add_complex, i)) for i in
                           range(m)]
        for i in range(m):
            btn_add_complex[i].pack(side=tk.TOP)

        def init_interactions():

            interactions = [(Interaction(
                inp_arg=np.array([int(ent_inp_values[i][j].get()) for j in range(n)]),
                out_arg=np.array(
                    [[int(ent_out_values[i][k][j].get()) for j in range(n)] for k in range(len(ent_out_values[i]))])
            )) for i in range(m)]

            frm_params_base = tk.CTkFrame(frm_main)
            frm_params_base.grid(row=2, column=0, sticky=tk.NSEW)

            # frm_params = VerticalScrolledFrame(frm_params_base)
            frm_params = tk.CTkScrollableFrame(frm_params_base, width=500, height=100)
            frm_params.configure(width=max(n, m) * 60)
            frm_params.pack(side=tk.TOP, expand=True, pady=1)

            frm_init_values = tk.CTkFrame(frm_params)
            frm_lam_base = tk.CTkFrame(frm_params)
            frm_lam = [tk.CTkFrame(frm_lam_base) for i in range(m)]
            frm_prob = {key: tk.CTkFrame(frm_params) for key in interactions_with_prob}
            frm_time = tk.CTkFrame(frm_params)
            frm_count = tk.CTkFrame(frm_params)
            frm_count_draw = tk.CTkFrame(frm_params)

            frm_init_values.pack()
            frm_lam_base.pack()
            for el in frm_lam:
                el.pack()
            for el in frm_prob.values():
                el.pack()
            frm_time.pack()
            frm_count.pack()
            frm_count_draw.pack()

            ent_init_values = [tk.CTkEntry(frm_init_values, width=8) for i in range(n)]
            ent_lam = [tk.CTkEntry(frm_lam[i], width=8) for i in range(m)]
            ent_prob = {key: [tk.CTkEntry(frm_prob[key], width=3) for i in range(count)]
                        for key, count in interactions_with_prob.items()}

            for i, el in enumerate(ent_init_values):
                tk.CTkLabel(frm_init_values, text='T' + str(i + 1) + ' ').pack(side=tk.LEFT)
                el.pack(side=tk.LEFT)

            for i, el in enumerate(ent_lam):
                tk.CTkLabel(frm_lam[i], text='lam' + str(i + 1) + '=').pack(side=tk.LEFT)
                el.pack(side=tk.LEFT)

            for key, ent in ent_prob.items():
                for i, el in enumerate(ent):
                    tk.CTkLabel(frm_prob[key], text='p' + str(key + 1) + str(i + 1) + '=').pack(side=tk.LEFT)
                    el.pack(side=tk.LEFT)

            ent_time = tk.CTkEntry(frm_time, width=4)
            ent_count = tk.CTkEntry(frm_count, width=4)
            ent_count_draw = tk.CTkEntry(frm_count_draw, width=4)

            tk.CTkLabel(frm_time, text='T=').pack(side=tk.LEFT)
            ent_time.pack(side=tk.LEFT)

            tk.CTkLabel(frm_count, text='N=').pack(side=tk.LEFT)
            ent_count.pack(side=tk.LEFT)

            tk.CTkLabel(frm_count, text='M=').pack(side=tk.LEFT)
            ent_count_draw.pack(side=tk.LEFT)

            def init_params():

                try:
                    init_values = np.array([int(ent_init_values[i].get()) for i in range(n)])
                    lam = np.array([float(ent_lam[i].get()) for i in range(m)])
                    time = int(ent_time.get())
                    probabilities = {num: np.array(list(map(lambda x: float(x.get()), values))) for num, values in
                                     ent_prob.items()}
                    count = int(ent_count.get())
                    count_draw = int(ent_count_draw.get())
                except Exception as ex:
                    print(ex)
                    return

                if any([False if abs(sum(val) - 1) < 1e-8 else True for val in probabilities.values()]):
                    return
                else:
                    for num, arr in probabilities.items():
                        interactions[num].add_probabilities(arr)

                mean, samples, trajectories_draw = modeling(interactions, init_values, lam, time,
                                                            n, m, count, count_draw)
                bins = int(1 + 3.32 * np.log10(count))

                combobox_var = tk.StringVar(value='T1')
                switch_var = tk.StringVar(value="trajectory")

                frm_plot_base = tk.CTkFrame(frm_main)
                frm_plot_switch = tk.CTkFrame(frm_plot_base)
                frm_plot = tk.CTkFrame(frm_plot_base)

                fig = Figure(figsize=(5, 5), dpi=100)
                # canvas = FigureCanvasTkAgg(fig, master=frm_plot)
                canvas = MyCanvas(fig, frm_plot, mean, samples, trajectories_draw,
                                  switch_var.get(), int(combobox_var.get()[-1])-1, time, count_draw, bins)

                tk.CTkLabel(frm_plot_switch, text='Диаграмма').pack(side=tk.LEFT)
                swt_plot = tk.CTkSwitch(frm_plot_switch,
                                        onvalue="trajectory", offvalue="diagram",
                                        text='Траектории', variable=switch_var,
                                        command=lambda: canvas.change_plot(mean, samples, trajectories_draw,
                                                                           switch_var.get(),
                                                                           int(combobox_var.get()[-1])-1,
                                                                           time, count_draw, bins)
                                        )
                swt_plot.pack(side=tk.LEFT)

                cmb_type = tk.CTkComboBox(frm_plot_switch,
                                          values=['T' + str(i + 1) for i in range(n)],
                                          variable=combobox_var,
                                          command=lambda x: canvas.change_plot(mean, samples, trajectories_draw,
                                                                             switch_var.get(),
                                                                             int(combobox_var.get()[-1])-1,
                                                                             time, count_draw, bins)
                                          )
                cmb_type.pack(side=tk.LEFT)

                frm_plot_switch.pack()
                frm_plot.pack(fill=tk.BOTH, expand=True)

                # change_plot(canvas, mean, samples, trajectories_draw,
                #             switch_var.get(), int(combobox_var.get()[-1]), time,
                #             count_draw,
                #             bins)

                toolbar = NavigationToolbar2Tk(canvas, frm_plot)
                toolbar.update()

                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                frm_plot_base.grid(row=0, column=1, rowspan=3, sticky=tk.NSEW)

            tk.CTkButton(frm_params_base, text='Рассчитать', command=init_params).pack()

        frm_btn_apply_complexes = tk.CTkFrame(frm_complexes_base)
        frm_btn_apply_complexes.pack()
        btn_apply_complexes = tk.CTkButton(frm_btn_apply_complexes, text='Применить', command=init_interactions)
        btn_apply_complexes.pack()

    btn_apply_sizes = tk.CTkButton(frm_sizes, text='Применить',
                                   command=lambda: init_frm_complexes(ent_n.get(), ent_m.get()))

    btn_apply_sizes.grid(row=2, column=1, columnspan=2)

    frm_main.mainloop()


if __name__ == '__main__':
    main()
