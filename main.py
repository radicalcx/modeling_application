import customtkinter as tk
from functools import partial

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from pandastable import Table
from scipy.stats import gaussian_kde
from datetime import datetime as dt
from os import makedirs
from os.path import isdir
from calculations import *
from statistics import *
from design import *

tk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
tk.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


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
        self.samples_without_norm = None
        self.samples = None
        self.trajectories_draw = None
        self.bins = None
        self.ent_bins = None
        self.bw = 0.3
        self.ent_bw = None
        self.combobox_var = None
        self.switch_var = None
        self.frm_plot_base = None
        self.frm_plot_switch = None
        self.frm_plot = None
        self.fig = None
        self.canvas = None
        # self.swt_plot = None
        self.cmb_plot = None
        self.cmb_type = None
        # self.df_chi2 = None
        self.df_stat = None
        # self.df_intervals = None
        self.toolbar = None
        self.folder_name = 'noname'
        self.frm_input = None
        self.ent_input = None

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
            self.mean, self.samples, self.trajectories_draw, self.samples_without_norm = modeling(
                self.interactions,
                self.init_values, self.lam,
                self.time,
                self.n, self.m, self.count, self.count_draw)
        except Exception as ex:
            show_exception(str(ex) + ' in modeling')
            return

        self.bins = int(1 + 3.32 * np.log10(self.count))

        self.combobox_var = tk.StringVar(value='T1')
        self.switch_var = tk.StringVar(value='Плотность распределения')

        self.frm_plot_base = tk.CTkFrame(self)
        self.frm_plot_switch = tk.CTkFrame(self.frm_plot_base, border_width=border_width_frm,
                                           border_color=border_color_frm,
                                           corner_radius=corner_radius_frm)
        self.frm_plot = tk.CTkFrame(self.frm_plot_base)

        self.fig = Figure(figsize=(1, 1), dpi=100)
        self.fig.subplots_adjust(top=0.96, bottom=0.068, right=0.999, left=0.068, wspace=0, hspace=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_plot)
        self.change_plot()

        # tk.CTkLabel(self.frm_plot_switch, text='Траектории', font=font_arg).pack(side=tk.LEFT, padx=20)
        # self.swt_plot = tk.CTkSwitch(self.frm_plot_switch,
        #                              onvalue="d", offvalue="t",
        #                              text='Диаграмма', variable=self.switch_var,
        #                              command=self.change_plot,
        #                              font=font_arg)

        # self.swt_plot.pack(side=tk.LEFT)
        self.cmb_plot = tk.CTkComboBox(self.frm_plot_switch,
                                       values=['Плотность распределения', 'Функция распределения', 'Траектории'],
                                       variable=self.switch_var,
                                       command=self.change_plot,
                                       font=font_arg)

        self.cmb_type = tk.CTkComboBox(self.frm_plot_switch,
                                       values=['T' + str(i + 1) for i in range(self.n)],
                                       variable=self.combobox_var,
                                       command=self.change_plot,
                                       font=font_arg)

        self.cmb_plot.pack(side=tk.LEFT, padx=padx_btn)
        self.cmb_type.pack(side=tk.LEFT, padx=padx_btn)

        self.frm_plot_switch.pack(padx=padx_frm, pady=1)

        try:
            # self.df_chi2, self.df_intervals = calculate_chi2(self.samples, self.bins, self.n, self.count)
            self.df_stat = calculate_statistic(self.samples, self.n, self.count, self.bins)
        except Exception as ex:
            show_exception(str(ex) + ' in calculate_statistic')
            return

        self.ent_bins = tk.CTkEntry(self.frm_plot_switch, width=35)
        self.ent_bins.insert(0, str(self.bins))
        self.ent_bins.bind('<Return>', self.change_bins)

        self.ent_bw = tk.CTkEntry(self.frm_plot_switch, width=45)
        self.ent_bw.insert(0, str(self.bw))
        self.ent_bw.bind('<Return>', self.change_bw)

        self.ent_bins.pack(side=tk.LEFT, padx=padx_btn)
        self.ent_bw.pack(side=tk.LEFT, padx=padx_btn)

        tk.CTkButton(self.frm_plot_switch, text='Сохранить',
                     font=font_arg, command=self.save_samples).pack(side=tk.RIGHT, padx=padx_btn, pady=pady_btn)

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
        # plot = self.fig.add_subplot()

        # plot = axisartist.Subplot(self.fig, 111)
        # self.fig.add_axes(plot)
        # plot.axis["bottom"].set_axisline_style("-|>", size=1.5)
        # plot.axis["left"].set_axisline_style("-|>", size=1.5)

        plot = self.fig.add_subplot()

        idx = int(self.combobox_var.get()[-1]) - 1
        plot_type = self.switch_var.get()[0]
        if plot_type == 'П':
            try:
                kde_pdf = gaussian_kde(self.samples[:, idx], bw_method=self.bw).pdf
            except Exception as ex:
                show_exception(str(ex) + ' in gaussian_kde')
                return

            plot.hist(self.samples[:, idx],
                      bins=self.bins,
                      range=(self.samples[:, idx].min(), self.samples[:, idx].max()),
                      density='True',
                      color='blue',
                      label='эмп. распр.')
            grid = np.arange(self.samples[:, idx].min(), self.samples[:, idx].max(), 0.1)
            plot.plot(grid, norm.pdf(grid, 0, 1), color='black', linewidth=3.0, label=r'плот. распр. $N(0, 1)$')
            plot.plot(grid, kde_pdf(grid), color='red', linewidth=3.0, linestyle='--', label='ЯОП')

            plot.set_xlabel(r'Значения случайной величины $\xi(T)$', fontsize=17, fontweight='regular', labelpad=7)
            plot.set_ylabel(r'Плотность вероятности', fontsize=17, fontweight='regular', labelpad=7)
            plot.set_title('Диаграмма распределения для Т' + str(idx + 1), fontsize=17, fontweight='bold')
            plot.legend(fontsize='xx-large')
        elif plot_type == 'Т':
            plot.plot(np.linspace(0, self.time, self.time * 4), self.mean[:, idx], color='red', linewidth=5,
                      zorder=self.count_draw + 1, linestyle='--', label=r'$A(t)$')
            for tr in self.trajectories_draw:
                plot.plot(tr.time, tr.track[:, idx], linewidth=2)
            plot.set_xlabel(r'$t$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_ylabel(r'$\xi(t)$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_title('Траектории для Т' + str(idx + 1), fontsize=17, fontweight='bold')
            plot.grid()
            plot.legend(fontsize='xx-large')
        else:
            x = np.sort(self.samples[:, idx])
            y_emp = 1. * np.arange(self.count) / (self.count - 1)
            y_thr = norm.cdf(x)
            plot.plot(x, y_thr, linewidth=3, label=r'$F(x)$ для $N(0, 1)$')
            plot.plot(x, y_emp, linewidth=3, label=r'$F_N(x)$')
            plot.grid(axis='y')
            plot.legend(fontsize='xx-large')
            plot.set_xlabel(r'$x$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_ylabel(r'$P(\xi<x)$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_title('Функция распределения для Т' + str(idx + 1), fontsize=17, fontweight='bold')

        self.canvas.draw()

    def change_bins(self, event=None):
        try:
            self.bins = int(self.ent_bins.get())
        except Exception as ex:
            show_exception(ex)
            return
        try:
            self.df_stat = calculate_statistic(self.samples, self.n, self.count, self.bins)
        except Exception as ex:
            show_exception(str(ex) + 'in calculate_statistic')
        self.change_plot()

    def change_bw(self, event=None):
        try:
            self.bw = float(self.ent_bw.get())
        except Exception as ex:
            show_exception(ex)
            return
        self.change_plot()

    def set_folder_name(self):
        name = self.ent_input.get()
        if isdir('results/' + name):
            show_exception('Folder with this name is already exist')
            return

        self.folder_name = name

        folder = 'results/' + name
        makedirs(folder)
        self.frm_input.destroy()
        with open(folder + '/parameters.txt', 'w+') as params_file:
            params_file.write('interactions:\n')
            for inter in self.interactions:
                params_file.write(inter.inp.__str__() + ' -> ')
                for el in inter.out:
                    params_file.write(el.__str__() + ' ')
                params_file.write('\n\n')
            params_file.write('init values:\n')
            params_file.write(self.init_values.__str__() + '\n\n')

            params_file.write('lambda:\n')
            params_file.write(self.lam.__str__() + '\n\n')

            params_file.write('probabilities:\n')
            for num, arr in self.probabilities.items():
                params_file.write(str(num + 1) + ': ' + arr.__str__() + '\n')
            params_file.write('\n')

            params_file.write('T: ' + str(self.time) + '\n')
            params_file.write('N: ' + str(self.count) + '\n')
            params_file.write('bins: ' + str(self.bins) + '\n')
            params_file.write('bw: ' + str(self.bw) + '\n\n')

            params_file.write('mean_emp:\n')
            params_file.write(self.samples_without_norm.mean(axis=0).__str__()+'\n\n')
            params_file.write('mean_thr:\n')
            params_file.write(self.mean[-1].__str__()+'\n\n')

            params_file.write('std:\n')
            params_file.write(self.samples_without_norm.std(axis=0).__str__()+'\n\n')

        self.df_stat.to_excel(folder + '/statistics.xlsx')
        np.savetxt(folder + '/samples.csv', self.samples_without_norm, fmt='%s')
        np.savetxt(folder + '/samples_norm.csv', self.samples, fmt='%s')

        folder += '/pictures'
        makedirs(folder)

        temp_fig = Figure(figsize=(10, 10), dpi=100)
        temp_fig.subplots_adjust(top=0.98, bottom=0.08, right=0.98, left=0.08, wspace=0, hspace=0)

        for idx in range(self.n):
            plot = temp_fig.add_subplot()

            kde_pdf = gaussian_kde(self.samples[:, idx], bw_method=self.bw).pdf
            plot.hist(self.samples[:, idx],
                      bins=self.bins,
                      range=(self.samples[:, idx].min(), self.samples[:, idx].max()),
                      density='True',
                      color='blue',
                      label='эмп. распр.')
            grid = np.arange(self.samples[:, idx].min(), self.samples[:, idx].max(), 0.1)
            plot.plot(grid, norm.pdf(grid, 0, 1), color='black', linewidth=3.0, label=r'плот. распр. $N(0, 1)$')
            plot.plot(grid, kde_pdf(grid), color='red', linewidth=3.0, linestyle='--', label='ЯОП')

            plot.set_xlabel(r'Значения случайной величины $\xi(T)$', fontsize=17, fontweight='regular', labelpad=7)
            plot.set_ylabel(r'Плотность вероятности', fontsize=17, fontweight='regular', labelpad=7)
            # plot.set_title('Диаграмма распределения для Т' + str(idx + 1), fontsize=17, fontweight='bold')
            plot.legend(fontsize='xx-large')

            temp_fig.savefig(fname=folder+'/T' + str(idx+1) + '_pdf.png')
            temp_fig.clf()

            plot = temp_fig.add_subplot()
            plot.plot(np.linspace(0, self.time, self.time * 4), self.mean[:, idx], color='red', linewidth=5,
                      zorder=self.count_draw + 1, linestyle='--', label=r'$A(t)$')
            for tr in self.trajectories_draw:
                plot.plot(tr.time, tr.track[:, idx], linewidth=2)
            plot.set_xlabel(r'$t$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_ylabel(r'$\xi(t)$', fontsize=17, fontweight='regular', labelpad=3)
            # plot.set_title('Траектории для Т' + str(idx + 1), fontsize=17, fontweight='bold')
            plot.grid()
            plot.legend(fontsize='xx-large')

            temp_fig.savefig(fname=folder + '/T' + str(idx+1) + '_trajectories.jpeg', format='jpeg')
            temp_fig.clf()

            plot = temp_fig.add_subplot()
            x = np.sort(self.samples[:, idx])
            y_emp = 1. * np.arange(self.count) / (self.count - 1)
            y_thr = norm.cdf(x)
            plot.plot(x, y_thr, linewidth=3, label=r'$F(x)$ для $N(0, 1)$')
            plot.plot(x, y_emp, linewidth=3, label=r'$F_N(x)$')
            plot.grid(axis='y')
            plot.legend(fontsize='xx-large')
            plot.set_xlabel(r'$x$', fontsize=17, fontweight='regular', labelpad=3)
            plot.set_ylabel(r'$P(\xi<x)$', fontsize=17, fontweight='regular', labelpad=3)
            # plot.set_title('Функция распределения для Т' + str(idx + 1), fontsize=17, fontweight='bold')

            temp_fig.savefig(fname=folder + '/T' + str(idx + 1) + '_cdf.jpeg')
            temp_fig.clf()

    def save_samples(self):
        self.frm_input = tk.CTkToplevel()
        self.frm_input.title('input file name')
        self.ent_input = tk.CTkEntry(self.frm_input, width=300)
        self.ent_input.pack(ipadx=10, ipady=10, padx=10, pady=10)
        tk.CTkButton(self.frm_input, text='OK',
                     font=font_arg, command=self.set_folder_name).pack(padx=padx_btn, pady=pady_btn)
        self.frm_input.resizable(width=False, height=False)
        self.frm_input.attributes("-topmost", True)
        self.frm_input.mainloop()

    def show_statistics(self):
        frm = tk.CTkToplevel()
        frm.title('statistics')
        frm.geometry('350x100')
        frm.resizable(width=False, height=False)
        frm.attributes("-topmost", True)

        # frm_scroll_base = tk.CTkScrollableFrame(frm)
        # frm_scroll_base.pack(expand=True, fill=tk.BOTH)
        # frm_chi2 = tk.CTkFrame(frm_scroll_base)
        frm_stat = tk.CTkFrame(frm)
        table_stat = Table(frm_stat, dataframe=self.df_stat,
                           showtoolbar=False, showstatusbar=False)
        # frm_chi2.pack(expand=True, fill=tk.BOTH)
        frm_stat.pack(expand=True, fill=tk.BOTH)
        table_stat.show()

        # for i in range(self.n):
        #     frm_temp_tb = tk.CTkFrame(frm_scroll_base)
        #     frm_temp_lb = tk.CTkFrame(frm_scroll_base)
        #     label = tk.CTkLabel(frm_temp_lb, text='T' + str(i + 1), font=font_arg)
        #     table = Table(frm_temp_tb, dataframe=self.df_intervals[i],
        #                   showtoolbar=False, showstatusbar=False)
        #     frm_temp_lb.pack(expand=True, fill=tk.BOTH)
        #     frm_temp_tb.pack(expand=True, fill=tk.BOTH)
        #     label.pack(expand=True, fill=tk.BOTH)
        #     table.show()

        frm.mainloop()


if __name__ == "__main__":
    app = App()
    app.mainloop()
