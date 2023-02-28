import tkinter as tk
import typing
from functools import partial
import numpy as np
import numpy.typing
from numpy.random import exponential, rand
import pandas as pd


class Interaction:
    def __init__(self, inp_arg, out_arg):
        self.inp = np.array(inp_arg)
        self.out = np.array(out_arg)
        self.prob = None

    def add_probabilities(self, prob_arg: numpy.typing.NDArray):
        self.prob = np.array([prob_arg[0]])
        for el in prob_arg[1:]:
            self.prob = np.append(self.prob, self.prob[-1] + el)

    def choose_out(self):
        if self.prob is None:
            return self.out[0]
        else:
            coin = rand()
            num = (coin < self.prob).argmax()
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
                time_array = np.append(time_array, time_array[-1]+t)
                break
        else:
            trajectory = np.append(trajectory, [trajectory[-1]], axis=0)
            time_array = np.append(time_array, time_array[-1] + 1)

    trajectory[-1] = trajectory[-2]
    time_array[-1] = time
    return trajectory, time_array


def main():
    frm_main = tk.Tk()
    frm_main.title('modeling_application')

    frm_sizes = tk.LabelFrame(frm_main, text='Размерности системы', padx=15, pady=10)
    frm_sizes.grid(row=0, column=0, padx=15, pady=15)

    lbl_n = tk.Label(frm_sizes, text='Количество типов элементов')
    lbl_m = tk.Label(frm_sizes, text='Количество комплексов взаимодействия')

    ent_n = tk.Entry(frm_sizes, width=2)
    ent_m = tk.Entry(frm_sizes, width=2)

    lbl_n.grid(row=0, column=0, padx=15, pady=5)
    ent_n.grid(row=0, column=1, padx=15, pady=5)

    lbl_m.grid(row=1, column=0, padx=15, pady=5)
    ent_m.grid(row=1, column=1, padx=15, pady=5)

    frm_complexes = None
    frm_complex_rows = None
    frm_out = None
    frm_out_text = None
    frm_inp_text = None
    frm_inp_btn = None

    ent_inp_values = None
    ent_out_values = None

    btn_add_complex = None
    btn_apply_complexes = None

    interactions = None
    probabilities = None
    time = None
    init_values = None
    lam = None

    interactions_with_prob = {}

    frm_params = None

    ent_init_values = None
    ent_lam = None
    ent_prob = None
    ent_time = None

    def init_frm_complexes(n_str, m_str):
        try:
            n = int(n_str)
            m = int(m_str)
        except Exception as ex:
            return

        nonlocal frm_complexes, frm_complex_rows
        nonlocal frm_out, frm_out_text
        nonlocal frm_inp_text, frm_inp_btn
        nonlocal interactions_with_prob
        if frm_complexes is not None:
            frm_complexes.destroy()
            interactions_with_prob.clear()

        frm_complexes = tk.LabelFrame(frm_main, padx=15, pady=10, text='Комплексы взаимодействия')
        frm_complex_rows = [tk.Frame(frm_complexes, padx=3, pady=0) for i in range(m)]
        frm_out = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]
        frm_out_text = [[tk.Frame(frm_out[i], padx=0, pady=0), ] for i in range(m)]
        frm_inp_text = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]
        frm_inp_btn = [tk.Frame(frm_complex_rows[i], padx=15, pady=5) for i in range(m)]

        frm_complexes.grid(row=1, column=0, padx=15, pady=15)
        for i in range(m):
            frm_complex_rows[i].pack(padx=15, pady=5)
            frm_inp_text[i].grid(row=0, column=0, padx=0)
            frm_inp_btn[i].grid(row=1, column=0, padx=0)
            frm_out[i].grid(row=0, column=1, padx=0)
            frm_out_text[i][0].pack(side=tk.TOP, padx=0, pady=0)

        nonlocal ent_inp_values, ent_out_values
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

        nonlocal btn_add_complex
        btn_add_complex = [tk.Button(frm_inp_btn[i], text='Добавить', command=partial(add_complex, i)) for i in
                           range(m)]
        for i in range(m):
            btn_add_complex[i].pack(side=tk.TOP)

        def init_interactions():
            nonlocal ent_inp_values
            nonlocal interactions
            nonlocal frm_params
            nonlocal ent_lam, ent_init_values, ent_prob
            nonlocal ent_time

            if interactions is not None:
                interactions = None

            interactions = [(Interaction(
                inp_arg=np.array([int(ent_inp_values[i][j].get()) for j in range(n)]),
                out_arg=np.array(
                    [[int(ent_out_values[i][k][j].get()) for j in range(n)] for k in range(len(ent_out_values[i]))])
            )) for i in range(m)]

            for inter in interactions:
                print(inter.inp, inter.out)

            if frm_params is not None:
                frm_params.destroy()

            frm_params = tk.LabelFrame(frm_main, text='Параметры системы')
            frm_params.grid(row=1, column=1)

            frm_init_values = tk.Frame(frm_params, padx=15, pady=10)
            frm_lam = tk.Frame(frm_params, padx=15, pady=10)
            frm_prob = {key: tk.Frame(frm_params, padx=15, pady=10) for key in interactions_with_prob}
            frm_time = tk.Frame(frm_params, padx=15, pady=10)

            frm_init_values.pack()
            frm_lam.pack()
            for el in frm_prob.values():
                el.pack()
            frm_time.pack()

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

            tk.Label(frm_time, text='T=').pack(side=tk.LEFT)
            ent_time.pack(side=tk.LEFT)

            def init_params():
                nonlocal time, probabilities, lam, init_values

                try:
                    init_values = np.array([int(ent_init_values[i].get()) for i in range(n)])
                    lam = np.array([float(ent_lam[i].get()) for i in range(m)])
                    time = int(ent_time.get())
                    probabilities = {num: np.array(list(map(lambda x: float(x.get()), values))) for num, values in
                                     ent_prob.items()}
                except Exception as ex:
                    print(ex)

                if any([False if abs(sum(val) - 1) < 1e-8 else True for val in probabilities.values()]):
                    return
                else:
                    for num, arr in probabilities.items():
                        interactions[num].add_probabilities(arr)

                tr, t = create_trajectory(interactions, init_values, lam, time, m)
                print(tr, t)

            tk.Button(frm_params, text='Рассчитать', command=init_params).pack(side=tk.TOP)

        nonlocal btn_apply_complexes
        btn_apply_complexes = tk.Button(frm_complexes, text='Применить', command=init_interactions)
        btn_apply_complexes.pack()

    btn_apply_params = tk.Button(frm_sizes, text='Применить',
                                 command=lambda: init_frm_complexes(ent_n.get(), ent_m.get()))

    btn_apply_params.grid(row=2, column=1, columnspan=2)

    frm_main.mainloop()


if __name__ == '__main__':
    main()
