import customtkinter as tk
import typing
from functools import partial
import numpy as np
import sympy as sp
import numpy.typing
from numpy.random import exponential, rand
import pandas as pd
from scipy.stats import norm, chi2
from scipy.integrate import odeint
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from pandastable import Table
from tkinter.font import Font

tk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
tk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"
border_color_frm = 'SpringGreen3'
border_width_frm = 3
corner_radius_frm = 10
font_arg = ('Adobe Garamond Pro Bold', 15)
padx_frm = 9
pady_frm = 16
ipadx_frm = 20
ipady_frm = 5
padx_lbl = 20
pady_lbl = 5
padx_btn = 20
pady_btn = 5


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


def calculate_expected_value(inter: typing.List[Interaction], init_val, lam, time, n, m):
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
    try:
        trajectories_draw = [create_trajectory(inter, init_val, lam, time, m) for i in range(M)]
    except Exception as ex:
        show_exception(str(ex) + ' in create_trajectory')
        return
    for i in range(M):
        samples[i] = trajectories_draw[i].track[-1]
    for i in range(M, N):
        samples[i] = create_trajectory(inter, init_val, lam, time, m).track[-1]

    mean = calculate_expected_value(inter, init_val, lam, time, n, m)
    std = samples.std(axis=0)

    for i in range(n):
        samples[:, i] = (samples[:, i] - mean[-1][i]) / (std[i] + 1e-10)

    return mean, samples, trajectories_draw


def calculate_chi2(sample: numpy.typing.NDArray, bins, n, N):
    intervals = [np.empty(bins + 1) for i in range(n)]
    emp_frequencies = [np.empty(bins) for i in range(n)]

    for i in range(n):
        emp_frequencies[i], intervals[i] = np.histogram(sample[:, i], bins=bins)

    for i in range(n):
        while emp_frequencies[i][0] < 5:
            temp = emp_frequencies[i][0]
            emp_frequencies[i] = emp_frequencies[i][1:]
            emp_frequencies[i][0] += temp
            try:
                intervals[i] = np.delete(intervals[i], 1)
            except Exception as ex:
                break

        while emp_frequencies[i][-1] < 5:
            temp = emp_frequencies[i][-1]
            emp_frequencies[i] = emp_frequencies[i][:-1]
            emp_frequencies[i][-1] += temp
            try:
                intervals[i] = np.delete(intervals[i], -2)
            except Exception as ex:
                break
    thr_frequencies = [np.empty(len(emp_frequencies[i])) for i in range(n)]

    for i in range(n):
        thr_frequencies[i][0] = norm(0, 1).cdf(intervals[i][1])

    for i in range(n):
        for j in range(1, emp_frequencies[i].shape[0] - 1):
            thr_frequencies[i][j] = norm(0, 1).cdf(intervals[i][j + 1]) - norm(0, 1).cdf(intervals[i][j])

    for i in range(n):
        thr_frequencies[i][-1] = 1 - norm(0, 1).cdf(intervals[i][-2])

    t = [el * N for el in thr_frequencies]

    table_chi2 = np.empty((n, 2))
    for i in range(n):
        table_chi2[i, 0] = (((emp_frequencies[i] - N * thr_frequencies[i]) ** 2) / (N * thr_frequencies[i])).sum()
        table_chi2[i, 1] = chi2(emp_frequencies[i].shape[0] - 1).ppf(0.95)

    df_intervals = [pd.DataFrame({'Эмперические': emp_frequencies[i], 'Теоретические': thr_frequencies[i] * N,
                                  'left': intervals[i][:-1], 'right': intervals[i][1:]}) for i in range(n)]

    return pd.DataFrame(table_chi2, columns=['Статистика', 'Квантиль']), df_intervals


def show_exception(exc: Exception | str):
    frm_ex = tk.CTkToplevel()
    frm_ex.title('error')
    tk.CTkLabel(frm_ex, text='error: ' + str(exc)).pack(ipadx=10, ipady=10, padx=10, pady=10)
    frm_ex.resizable(width=False, height=False)
    frm_ex.attributes("-topmost", True)
    frm_ex.mainloop()


class App(tk.CTk):
    def __init__(self):
        super().__init__()
        self.title('modeling_application')
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=10)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure((1, 2), weight=1)

        self.frm_sizes = tk.CTkFrame(self, border_width=border_width_frm, border_color=border_color_frm,
                                     corner_radius=corner_radius_frm)
        self.frm_sizes.grid_columnconfigure((0, 1), weight=1)
        self.frm_sizes.grid_rowconfigure((0, 1, 2), weight=0)
        self.frm_sizes.grid(row=0, column=0, sticky=tk.NSEW,
                            padx=padx_frm,
                            pady=pady_frm,
                            ipadx=ipadx_frm,
                            ipady=ipady_frm
                            )

        self.lbl_n = tk.CTkLabel(self.frm_sizes, text='Количество типов элементов', font=font_arg)
        self.lbl_m = tk.CTkLabel(self.frm_sizes, text='Количество комплексов взаимодействия', font=font_arg)

        self.ent_n = tk.CTkEntry(self.frm_sizes, width=28)
        self.ent_m = tk.CTkEntry(self.frm_sizes, width=28)

        self.lbl_n.grid(row=0, column=0, padx=padx_lbl, pady=pady_lbl)
        self.ent_n.grid(row=0, column=1, padx=padx_lbl, pady=pady_lbl)

        self.lbl_m.grid(row=1, column=0)
        self.ent_m.grid(row=1, column=1)
        self.btn_apply_sizes = tk.CTkButton(self.frm_sizes, text='Применить',
                                            command=self.init_frm_complexes,
                                            font=font_arg)

        self.btn_apply_sizes.grid(row=2, column=1)

        self.n = None
        self.m = None
        self.interactions_with_prob = None
        self.frm_complexes_base = None
        self.frm_complexes = None
        self.frm_complex_rows = None
        self.frm_out = None
        self.frm_out_text = None
        self.frm_inp_text = None
        self.frm_inp_btn = None
        self.ent_inp_values = None
        self.ent_out_values = None
        self.btn_add_complex = None
        self.frm_btn_apply_complexes = None
        self.btn_apply_complexes = None
        self.interactions = None
        self.frm_params_base = None
        self.frm_params = None
        self.frm_init_values = None
        self.frm_lam_base = None
        self.frm_time = None
        self.frm_lam = None
        self.frm_prob = None
        self.frm_count = None
        self.frm_count_draw = None
        self.ent_init_values = None
        self.ent_lam = None
        self.ent_prob = None
        self.ent_time = None
        self.ent_count = None
        self.ent_count_draw = None
        self.init_values = None
        self.lam = None
        self.time = None
        self.probabilities = None
        self.count = None
        self.count_draw = None
        self.mean = None
        self.samples = None
        self.trajectories_draw = None
        self.bins = None
        self.combobox_var = None
        self.switch_var = None
        self.frm_plot_base = None
        self.frm_plot_switch = None
        self.frm_plot = None
        self.fig = None
        self.canvas = None
        self.swt_plot = None
        self.cmb_type = None
        self.df_chi2 = None
        self.df_intervals = None
        self.toolbar = None

    def init_frm_complexes(self):
        try:
            self.n = int(self.ent_n.get())
            self.m = int(self.ent_m.get())
        except Exception as ex:
            show_exception(ex)
            return

        self.interactions_with_prob = {}

        self.frm_complexes_base = tk.CTkFrame(self, border_width=border_width_frm, border_color=border_color_frm,
                                              corner_radius=corner_radius_frm, )
        self.frm_complexes = tk.CTkScrollableFrame(self.frm_complexes_base, width=500)
        self.frm_complex_rows = [tk.CTkFrame(self.frm_complexes) for i in range(self.m)]
        self.frm_out = [tk.CTkFrame(self.frm_complex_rows[i]) for i in range(self.m)]
        self.frm_out_text = [[tk.CTkFrame(self.frm_out[i]), ] for i in range(self.m)]
        self.frm_inp_text = [tk.CTkFrame(self.frm_complex_rows[i]) for i in range(self.m)]
        self.frm_inp_btn = [tk.CTkFrame(self.frm_complex_rows[i]) for i in range(self.m)]

        self.frm_complexes_base.grid(row=1, column=0, sticky=tk.NSEW,
                                     padx=padx_frm,
                                     pady=pady_frm,
                                     ipadx=ipadx_frm,
                                     ipady=ipady_frm
                                     )
        tk.CTkLabel(self.frm_complexes_base, text='Комплексы взаимодействия', font=font_arg).pack(padx=padx_lbl,
                                                                                                  pady=pady_lbl)
        self.frm_complexes.pack(expand=True, fill=tk.BOTH, padx=padx_frm)

        for i in range(self.m):
            self.frm_complex_rows[i].pack(padx=15, pady=5)
            self.frm_inp_text[i].grid(row=0, column=0, padx=0, sticky=tk.NSEW)
            self.frm_inp_btn[i].grid(row=1, column=0, padx=0, sticky=tk.NSEW)
            self.frm_out[i].grid(row=0, column=1, padx=0, sticky=tk.NSEW)
            self.frm_out_text[i][0].pack(side=tk.TOP, padx=0, pady=0, expand=True)

        self.ent_inp_values = [[tk.CTkEntry(self.frm_inp_text[i], width=1) for j in range(self.n)] for i in
                               range(self.m)]
        self.ent_out_values = [[[tk.CTkEntry(self.frm_out_text[i][0], width=1) for j in range(self.n)], ] for i in
                               range(self.m)]
        self.px = 0
        self.py = 3
        for i in range(self.m):
            for j in range(self.n - 1):
                self.ent_inp_values[i][j].pack(side=tk.LEFT, padx=self.px, pady=self.py)
                self.ent_inp_values[i][j].insert(0, '0')
                tk.CTkLabel(self.frm_inp_text[i], text='T' + chr(8321 + j) + '+', font=font_arg).pack(side=tk.LEFT,
                                                                                                      padx=self.px,
                                                                                                      pady=self.py)
            self.ent_inp_values[i][self.n - 1].pack(side=tk.LEFT, padx=self.px, pady=self.py)
            self.ent_inp_values[i][self.n - 1].insert(0, '0')
            tk.CTkLabel(self.frm_inp_text[i], text='T' + chr(8321 + self.n - 1) + ' \N{RIGHTWARDS BLACK ARROW}',
                        font=font_arg).pack(side=tk.LEFT, padx=self.px, pady=self.py)

        for i in range(self.m):
            for j in range(self.n - 1):
                self.ent_out_values[i][0][j].pack(side=tk.LEFT, padx=self.px, pady=self.py)
                self.ent_out_values[i][0][j].insert(0, '0')
                tk.CTkLabel(self.frm_out_text[i][0], text='T' + chr(8321 + j) + '+', font=font_arg).pack(side=tk.LEFT,
                                                                                                         padx=self.px,
                                                                                                         pady=self.py)
            self.ent_out_values[i][0][self.n - 1].pack(side=tk.LEFT, padx=self.px, pady=self.py)
            self.ent_out_values[i][0][self.n - 1].insert(0, '0')
            tk.CTkLabel(self.frm_out_text[i][0], text='T' + chr(8321 + self.n - 1), font=font_arg).pack(side=tk.LEFT,
                                                                                                        padx=self.px,
                                                                                                        pady=self.py)

        self.btn_add_complex = [
            tk.CTkButton(self.frm_inp_btn[i], text='Добавить', font=font_arg, command=partial(self.add_complex, i))
            for i in
            range(self.m)]
        for i in range(self.m):
            self.btn_add_complex[i].pack(side=tk.TOP)

        # self.frm_btn_apply_complexes = tk.CTkFrame(self.frm_complexes_base)
        # self.frm_btn_apply_complexes.pack()
        self.btn_apply_complexes = tk.CTkButton(self.frm_complexes_base, text='Применить',
                                                command=self.init_interactions,
                                                font=font_arg, )
        self.btn_apply_complexes.pack(pady=pady_btn)

    def add_complex(self, idx):
        self.frm_out_text[idx].append(tk.CTkFrame(self.frm_out[idx]))
        self.frm_out_text[idx][-1].pack()

        self.ent_out_values[idx].append([tk.CTkEntry(self.frm_out_text[idx][-1], width=1) for j in range(self.n)])
        for j in range(self.n - 1):
            self.ent_out_values[idx][-1][j].pack(side=tk.LEFT, padx=self.px, pady=self.py)
            self.ent_out_values[idx][-1][j].insert(0, '0')
            tk.CTkLabel(self.frm_out_text[idx][-1], text='T' + chr(8321 + j) + '+', font=font_arg).pack(side=tk.LEFT,
                                                                                                        padx=self.px,
                                                                                                        pady=self.py)
        self.ent_out_values[idx][-1][self.n - 1].pack(side=tk.LEFT, padx=self.px, pady=self.py)
        self.ent_out_values[idx][-1][self.n - 1].insert(0, '0')
        tk.CTkLabel(self.frm_out_text[idx][-1], text='T' + chr(8321 + self.n - 1), font=font_arg).pack(side=tk.LEFT,
                                                                                                       padx=self.px,
                                                                                                       pady=self.py)

        self.interactions_with_prob[idx] = self.interactions_with_prob.setdefault(idx, 1) + 1

    def init_interactions(self):
        try:
            self.interactions = [(Interaction(
                inp_arg=np.array([int(self.ent_inp_values[i][j].get()) for j in range(self.n)]),
                out_arg=np.array(
                    [[int(self.ent_out_values[i][k][j].get()) for j in range(self.n)] for k in
                     range(len(self.ent_out_values[i]))])
            )) for i in range(self.m)]
        except Exception as ex:
            show_exception(ex)
            return

        self.frm_params_base = tk.CTkFrame(self, border_width=border_width_frm, border_color=border_color_frm,
                                           corner_radius=corner_radius_frm)
        self.frm_params_base.grid(row=2, column=0, sticky=tk.NSEW,
                                  padx=padx_frm,
                                  pady=pady_frm,
                                  ipadx=ipadx_frm,
                                  ipady=ipady_frm
                                  )

        tk.CTkLabel(self.frm_params_base, text='Параметры системы', font=font_arg).pack(padx=padx_lbl,
                                                                                        pady=pady_lbl)

        self.frm_params = tk.CTkScrollableFrame(self.frm_params_base, height=100, )
        self.frm_params.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=padx_frm)

        self.frm_init_values = tk.CTkFrame(self.frm_params)
        # self.frm_lam_base = tk.CTkFrame(self.frm_params)
        self.frm_lam = [tk.CTkFrame(self.frm_params) for i in range(self.m)]
        self.frm_prob = {key: tk.CTkFrame(self.frm_params) for key in self.interactions_with_prob}
        self.frm_time = tk.CTkFrame(self.frm_params)
        self.frm_count = tk.CTkFrame(self.frm_params)
        self.frm_count_draw = tk.CTkFrame(self.frm_params)

        self.frm_init_values.pack(pady=self.py)
        # self.frm_lam_base.pack()
        for el in self.frm_lam:
            el.pack(pady=self.py)
        for el in self.frm_prob.values():
            el.pack(pady=self.py)
        self.frm_time.pack(pady=self.py)
        self.frm_count.pack(pady=self.py)
        self.frm_count_draw.pack(pady=self.py)

        self.ent_init_values = [tk.CTkEntry(self.frm_init_values, width=60) for i in range(self.n)]
        self.ent_lam = [tk.CTkEntry(self.frm_lam[i], width=100) for i in range(self.m)]
        self.ent_prob = {key: [tk.CTkEntry(self.frm_prob[key], width=50) for i in range(count)]
                         for key, count in self.interactions_with_prob.items()}

        for i, el in enumerate(self.ent_init_values):
            el.pack(side=tk.LEFT)
            tk.CTkLabel(self.frm_init_values, text='T' + chr(8321 + i) + ' ', font=font_arg).pack(side=tk.LEFT)

        for i, el in enumerate(self.ent_lam):
            tk.CTkLabel(self.frm_lam[i], text='\u03bb' + chr(8321 + i) + '= ', font=font_arg).pack(side=tk.LEFT)
            el.pack(side=tk.LEFT)

        for key, ent in self.ent_prob.items():
            for i, el in enumerate(ent):
                tk.CTkLabel(self.frm_prob[key], text='  p' + chr(8321 + key) + chr(8321 + i) + '= ',
                            font=font_arg).pack(
                    side=tk.LEFT)
                el.pack(side=tk.LEFT)

        self.ent_time = tk.CTkEntry(self.frm_time, width=50)
        self.ent_count = tk.CTkEntry(self.frm_count, width=50)
        self.ent_count_draw = tk.CTkEntry(self.frm_count_draw, width=50)

        tk.CTkLabel(self.frm_time, text='Время моделирования T = ', font=font_arg).pack(side=tk.LEFT)
        self.ent_time.pack(side=tk.LEFT)

        tk.CTkLabel(self.frm_count, text='Количество траекторий N = ', font=font_arg).pack(side=tk.LEFT)
        self.ent_count.pack(side=tk.LEFT)

        tk.CTkLabel(self.frm_count_draw, text='Количество траекторий на печать M = ', font=font_arg).pack(side=tk.LEFT)
        self.ent_count_draw.pack(side=tk.LEFT)

        tk.CTkButton(self.frm_params_base, text='Рассчитать', command=self.init_params, font=font_arg). \
            pack(pady=pady_btn)

    def init_params(self):
        try:
            self.init_values = np.array([int(self.ent_init_values[i].get()) for i in range(self.n)])
            self.lam = np.array([float(self.ent_lam[i].get()) for i in range(self.m)])
            self.time = int(self.ent_time.get())
            self.probabilities = {num: np.array(list(map(lambda x: float(x.get()), values))) for num, values in
                                  self.ent_prob.items()}
            self.count = int(self.ent_count.get())
            self.count_draw = int(self.ent_count_draw.get())
        except Exception as ex:
            show_exception(ex)
            return

        if any([False if abs(sum(val) - 1) < 1e-8 else True for val in self.probabilities.values()]):
            show_exception('the sum of the probabilities is not equal to one')
            return
        else:
            for num, arr in self.probabilities.items():
                self.interactions[num].add_probabilities(arr)

        try:
            self.mean, self.samples, self.trajectories_draw = modeling(self.interactions, self.init_values, self.lam,
                                                                       self.time,
                                                                       self.n, self.m, self.count, self.count_draw)
        except Exception as ex:
            show_exception(str(ex) + ' in modeling')
            return

        self.bins = int(1 + 3.32 * np.log10(self.count))

        self.combobox_var = tk.StringVar(value='T1')
        self.switch_var = tk.StringVar(value="trajectory")

        self.frm_plot_base = tk.CTkFrame(self)
        self.frm_plot_switch = tk.CTkFrame(self.frm_plot_base, border_width=border_width_frm,
                                           border_color=border_color_frm,
                                           corner_radius=corner_radius_frm)
        self.frm_plot = tk.CTkFrame(self.frm_plot_base)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_plot)
        self.change_plot()

        tk.CTkLabel(self.frm_plot_switch, text='Диаграмма  ', font=font_arg).pack(side=tk.LEFT, padx=20)
        self.swt_plot = tk.CTkSwitch(self.frm_plot_switch,
                                     onvalue="trajectory", offvalue="diagram",
                                     text='Траектории', variable=self.switch_var,
                                     command=self.change_plot,
                                     font=font_arg)

        self.swt_plot.pack(side=tk.LEFT)

        self.cmb_type = tk.CTkComboBox(self.frm_plot_switch,
                                       values=['T' + str(i + 1) for i in range(self.n)],
                                       variable=self.combobox_var,
                                       command=self.change_plot,
                                       font=font_arg)
        self.cmb_type.pack(side=tk.LEFT, padx=padx_btn)

        self.frm_plot_switch.pack(padx=padx_frm, pady=1)

        try:
            self.df_chi2, self.df_intervals = calculate_chi2(self.samples, self.bins, self.n, self.count)
        except Exception as ex:
            show_exception(str(ex) + ' in calculate_chi2')
            return

        tk.CTkButton(self.frm_plot_switch, text='Статистика',
                     font=font_arg, command=self.show_statistics).pack(side=tk.RIGHT, padx=padx_btn, pady=pady_btn)
        self.frm_plot.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frm_plot)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frm_plot_base.grid(row=0, column=1, rowspan=3, sticky=tk.NSEW,
                                padx=padx_frm,
                                pady=pady_frm,
                                ipadx=ipadx_frm,
                                ipady=ipady_frm
                                )

    def change_plot(self, arg=None):
        self.fig.clf()
        plot = self.fig.add_subplot()
        idx = int(self.combobox_var.get()[-1]) - 1
        if self.switch_var.get() == 'diagram':
            plot.hist(self.samples[:, idx],
                      bins=self.bins,
                      range=(self.samples[:, idx].min(), self.samples[:, idx].max()),
                      density='True',
                      color='blue')
            grid = np.arange(-3, 3, 0.1)
            plot.plot(grid, norm.pdf(grid, 0, 1), color='black', linewidth=3.0)
            plot.grid()
        elif self.switch_var.get() == 'trajectory':
            plot.plot(np.linspace(0, self.time, self.time), self.mean[:, idx], color='red', linewidth=5,
                      zorder=self.count_draw + 1, linestyle='--', label='expected value')
            for tr in self.trajectories_draw:
                plot.plot(tr.time, tr.track[:, idx], linewidth=2)
            plot.grid()
            plot.legend()
        else:
            return
        self.canvas.draw()

    def show_statistics(self):
        frm = tk.CTkToplevel()
        frm.title('statistics')
        frm.geometry('400x280')
        frm.resizable(width=False, height=False)
        frm.attributes("-topmost", True)

        frm_scroll_base = tk.CTkScrollableFrame(frm)
        frm_scroll_base.pack(expand=True, fill=tk.BOTH)
        frm_chi2 = tk.CTkFrame(frm_scroll_base)
        table_chi2 = Table(frm_chi2, dataframe=self.df_chi2,
                           showtoolbar=False, showstatusbar=False)
        frm_chi2.pack(expand=True, fill=tk.BOTH)
        table_chi2.show()

        for i in range(self.n):
            frm_temp_tb = tk.CTkFrame(frm_scroll_base)
            frm_temp_lb = tk.CTkFrame(frm_scroll_base)
            label = tk.CTkLabel(frm_temp_lb, text='T' + str(i + 1), font=font_arg)
            table = Table(frm_temp_tb, dataframe=self.df_intervals[i],
                          showtoolbar=False, showstatusbar=False)
            frm_temp_lb.pack(expand=True, fill=tk.BOTH)
            frm_temp_tb.pack(expand=True, fill=tk.BOTH)
            label.pack(expand=True, fill=tk.BOTH)
            table.show()

        frm.mainloop()


if __name__ == "__main__":
    app = App()
    app.mainloop()
