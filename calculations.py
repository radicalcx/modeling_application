import numpy as np
from numpy.typing import NDArray
from typing import List
from sympy import Symbol, prod, diff
from numpy.random import exponential, rand
from scipy.integrate import odeint
from main import show_exception


class Interaction:
    def __init__(self, inp_arg, out_arg):
        self.inp = np.array(inp_arg)
        self.out = np.array(out_arg)
        self.prob_intervals = [1, ]
        self.prob = [1, ]

    def add_probabilities(self, prob_arg: NDArray):
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


def create_trajectory(inter: List[Interaction], init_val, lam, time, m):
    trajectory = np.array([init_val])
    time_array = np.zeros(1)
    while time_array[-1] < time:
        temp_lam = [inter[i].inp * trajectory[-1] for i in range(m)]
        # tau = pd.Series(
        #     [exponential(1 / (np.prod(temp_lam[i][temp_lam[i] > 0]) * lam[i]))] for i in range(m)).sort_values()
        tau = sorted(
            [(i, exponential(1 / (np.prod(temp_lam[i][temp_lam[i] > 0]) * lam[i]))) for i in range(m)],
            key=lambda el: el[1])

        for num, t in tau:
            if np.greater_equal(trajectory[-1], inter[num].inp).all():
                trajectory = np.append(trajectory, [trajectory[-1] - inter[num].inp + inter[num].choose_out()], axis=0)
                time_array = np.append(time_array, time_array[-1] + t)
                break
        else:
            trajectory = np.append(trajectory, [trajectory[-1]], axis=0)
            time_array = np.append(time_array, time_array[-1] + 1)
            break

    trajectory[-1] = trajectory[-2]
    time_array[-1] = time

    return Trajectory(trajectory, time_array)


def calculate_expected_value(inter: List[Interaction], init_val, lam, time, n, m):
    s = [Symbol('s' + str(i)) for i in range(n)]
    arg_for_subs = [(s[i], 1) for i in range(n)]
    components = [sum(
        [prod(
            [s[j] ** inter[i].out[k][j] for j in range(n)]) * inter[i].prob[k] for k in range(len(inter[i].out))]) -
                  prod(
                      [s[j] ** inter[i].inp[j] for j in range(n)]) for i in range(m)]

    derivatives = [inter[i].inp.copy() for i in range(m)]

    diff_components = [[diff(components[i], s[j]) for j in range(n)] for i in range(m)]
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

    return odeint(system, init_val, np.linspace(0, time, time * 4), args=(lam,))


def modeling(inter: List[Interaction], init_val, lam, time, n, m, N, M):
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
    samples_norm = samples - mean[-1]
    std = np.sqrt((samples_norm ** 2).sum(axis=0) / N) + 1e-8
    samples_norm = samples_norm / std

    return mean, samples_norm, trajectories_draw, samples
