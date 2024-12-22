#!/usr/bin/env python3

import csv
import math

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy as scp

from mpl_toolkits.axes_grid1 import make_axes_locatable


# @nb.njit
def calc_ideal_gas_square_sound_speed_field(
        conserved_var_vec_arr, gamma, q):
    return (gamma
            * ideal_eos_p(conserved_var_vec_arr, gamma, q)
            / conserved_var_vec_arr[:, 0])


def ideal_eos_primitive(flow_var_vec, gamma: float = 1.4, q: float = 0.):
    # p = rho e (gam - 1)
    # e = p/(rho (gam - 1)) - q lam
    # E = e + (u**2)*0.5

    return (flow_var_vec[2] / (gamma - 1.) / flow_var_vec[0]
            + flow_var_vec[1]**2 * 0.5
            - flow_var_vec[3] * q)


def ideal_gas_eos_p(conserved_vars,
                    gamma: float = 1.4) -> float:
    # rho, rhou, rhoE = conserved_vars
    rho = conserved_vars[..., 0]
    rhou = conserved_vars[..., 1]
    rhoE = conserved_vars[..., 2]
    rhoekin = (0.5 * rhou**2 / rho)
    rhoeint = rhoE - rhoekin

    # p
    return (gamma - 1.) * rhoeint


# @nb.njit  # ('f8[:](f8[::1], f8, f8)')
def ideal_eos_p(flow_var_vec, gamma: float, q: float):
    """Return pressure field array corresponding to
        a given conserved variable vector field.

        Args:
            flow_var_vec: conserved-variable vector field (of dim (N, 4))
            gamma: heat capacity ratio
            q: specific heat release
        Returns:
            pressure field of dim (N,)
    """
# def ideal_eos_p(flow_var_vec, gamma: float = 1.4, q: float = 0., res):
    # p = rho e (gam - 1)
    # e = p/(rho (gam - 1)) - q lam
    # e + q lam = p/(rho (gam - 1))
    # (rho e + rho lam q)(gam - 1) = p
    rhoeint = (flow_var_vec[:, 2]
               - 0.5 * flow_var_vec[:, 1]**2 / flow_var_vec[:, 0])

    return (rhoeint + q*flow_var_vec[:, 3]) * (gamma - 1.)


def ideal_gas_compute_isentropic_bulk_modulus(conserved_vars,
                                              gamma: float = 1.4) -> float:
    return gamma * ideal_gas_eos_p(conserved_vars, gamma)


def ideal_gas_sound_speed(conserved_vars,
                          gamma: float = 1.4) -> float:
    rho = conserved_vars[..., 0]
    return np.abs(ideal_gas_compute_isentropic_bulk_modulus(conserved_vars, gamma)
                  / rho) ** 0.5


def conservative_from_primitive_singlephase(flow_var_vec, eos):
    res = np.zeros_like(flow_var_vec)

    start = 0
    # total_specific_energy = eos(flow_var_vec[i*4:(i+1)*4])
    # total_specific_energy = eos(flow_var_vec) / flow_var_vec[..., 0]
    total_energy = eos(flow_var_vec)
    res[..., start+0] = flow_var_vec[..., 0] * 1.
    res[..., start+1] = flow_var_vec[..., 0] * flow_var_vec[..., 1]
    # res[..., start+2] = flow_var_vec[..., 0] * total_specific_energy
    res[..., start+2] = total_energy
    # res[..., start+3] = flow_var_vec[..., -1]  # this remains as non-conserved

    return res


# def conservative_from_primitive(flow_var_vec, eoses):
#     # n_phases = len(eoses)
#     res = np.zeros_like(flow_var_vec)
#     for i, eos in enumerate(eoses):
#         start = i*4
#         end = start + 4
#         current_flow_var_vec = flow_var_vec[..., start:end]
#         # total_specific_energy = eos(flow_var_vec[i*4:(i+1)*4])
#         total_specific_energy = eos(current_flow_var_vec[..., :-1]) / current_flow_var_vec[..., 0]
#         res[..., start+0] = current_flow_var_vec[..., 0] * 1
#         res[..., start+1] = current_flow_var_vec[..., 0] * current_flow_var_vec[..., 1]
#         res[..., start+2] = current_flow_var_vec[..., 0] * total_specific_energy
#         res[..., start+3] = current_flow_var_vec[..., -1]  # this remains as non-conserved
#
#     return res
def conservative_from_primitive(flow_var_vec, gamma=1.4, q=0.):
    total_specific_energy = ideal_eos_primitive(flow_var_vec, gamma, q)

    return np.array([
        flow_var_vec[0] * 1.,
        flow_var_vec[0] * flow_var_vec[1],
        flow_var_vec[0] * total_specific_energy,
        flow_var_vec[0] * flow_var_vec[3]
    ])


@nb.vectorize('f8(f8, f8, f8, f8, f8, f8)')
def ideal_reactive_gas_total_energy_eos(rho, v, p, lam, q, gamma):
    return p/rho/(gamma - 1.) + (v**2)*0.5 - lam*q


@nb.vectorize('f8(f8, f8, f8, f8, f8, f8)')
def ideal_reactive_gas_pressure_eos(rho, v, E, lam, q, gamma):
    return (E - (v**2)*0.5 + lam*q) * rho * (gamma - 1.)


@nb.vectorize('f8(f8, f8, f8)')
def ideal_gas_sound_speed(rho, p, gamma):
    return (gamma * p / rho) ** 0.5


@nb.njit('f8(f8, f8)')
def chapman_jouget_detonation_speed(q, gamma):
    const = 0.5 * (gamma**2 - 1.) * q
    # const = mp.mpf(0.5) * (gamma**2 - 1.) * q
    return (gamma + const)**0.5 + const**0.5


@nb.vectorize('f8(f8, f8, f8, f8, f8)')
def arrhenius_reaction_rate(
        reaction_progress_variable,
        activation_energy,
        pressure,
        density,
        frequency_preexponential_factor):
    return (frequency_preexponential_factor
            * (1. - reaction_progress_variable)
            * np.exp(-activation_energy * density / pressure))


class TickRedrawer(matplotlib.artist.Artist):
    """Artist to redraw ticks."""

    __name__ = "ticks"

    zorder = 10

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer: matplotlib.backend_bases.RendererBase) -> None:
        """Draw the ticks."""
        if not self.get_visible():
            self.stale = False
            return

        renderer.open_group(self.__name__, gid=self.get_gid())

        for axis in (self.axes.xaxis, self.axes.yaxis):
            loc_min, loc_max = axis.get_view_interval()

            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_visible() and loc_min <= tick.get_loc() <= loc_max:
                    for artist in (tick.tick1line, tick.tick2line):
                        artist.draw(renderer)

        renderer.close_group(self.__name__)
        self.stale = False


def plot_reactive_euler_solution_profiles(
        solution_field,
        params,
        annotate_shock: bool = False,
        savefig: bool = False,
        type: str = 'svg',
        two_cols: bool = False,
        plot_reference: bool = False,
        figname=None,
        useOffset: bool = False,
        what_to_plot=[
            'pressure',
            'density',
            'velocity',
            'reaction_progress_variable',
            # 'internal_specific_energy',
            'total_specific_energy',
            # 'Mach_number',
            # 'reaction_rate'
            'sound_speed'
        ],
        reference=None,
        k=None,
        use_black: bool = True,
        plot_grid: bool = True,
        coordinate_text: str = r'Spatial coordinate $ x $',
        plot_title: bool = True,
        add_subtitles: bool = True) -> np.ndarray:
    """Plot 1D ideal reactive Euler equation solution profiles.

        Args:
            solution_field: field struct
            params: equation parameters dict / struct
            annotate_shock: Whether to put an annotation in the
                plot with the value at the shock rounded to 3
                decimal digits.
            savefig: Whether to save the plot as type.
            type: Plot picture format for matplotlib savefig (svg, png...).
            two_cols: Do weird two-col of squares format or not.
            plot_reference: Try to access and also plot
                `solution_field['reference_solution']`.
            figname: optional prepended name for the file and title
            useOffset: useOffset value in profile plots
            what_to_plot: list of names of variable fields, profiles
                of which are to be plotted
            reference: reference field struct (use `reference_solution`
                in solution_field struct if None)
            k: preexponential scaling factor for reaction rate

        Returns:
            solution_field

        Side-effects:
            Plot solution
    """

    res = solution_field['solution'][0]
    # assert np.all(np.isclose(res[:, 0], 0., atol=1e-15))
    # assert np.all(res[:, 0] > 0.)
    if plot_reference:
        if reference is None:
            ref = solution_field['reference_solution'][0]
        else:
            ref = reference['solution'][0]
    #     rs = (res, ref)
    # else:
    #     rs = (res,)

    # for r in rs:
    res_rho = res[:, 0]
    res_v = res[:, 1] / res_rho
    res_E = res[:, 2] / res_rho
    res_lambda = res[:, 3] / res_rho
    xi = solution_field['coordinate_mesh'][0]
    if reference is None:
        xi_ref = xi
    else:
        xi_ref = reference['coordinate_mesh'][0]

    if plot_reference:
        ref_rho = ref[:, 0]
        ref_v = ref[:, 1] / ref_rho
        ref_E = ref[:, 2] / ref_rho
        ref_lambda = ref[:, 3] / ref_rho

    specific_heat_release = params['specific_heat_release']
    heat_capacity_ratio_gamma = params['heat_capacity_ratio']
    activation_energy = params['activation_energy']
    res_p = ideal_reactive_gas_pressure_eos(
        res_rho,
        res_v,
        res_E,
        res_lambda,
        specific_heat_release,
        heat_capacity_ratio_gamma)
    res_c_s = ideal_gas_sound_speed(
        res_rho, res_p, heat_capacity_ratio_gamma)

    detonation_speed = chapman_jouget_detonation_speed(
        specific_heat_release, heat_capacity_ratio_gamma)

    # ys = [res_p, res_rho, res_v, res_lambda, res_E]
    ys = []
    labels = []
    long_labels = []

    for item in what_to_plot:
        if item == 'pressure':
            ys.append(res_p)
            labels.append(r'$ p $')
            long_labels.append(r'Pressure $ p $')
        elif item == 'density':
            ys.append(res_rho)
            labels.append(r'$ \rho $')
            long_labels.append(r'Density $ \rho $')
        elif item == 'velocity':
            ys.append(res_v)
            labels.append(r'$ v $')
            long_labels.append(r'Flow Speed $ v $ (Lab. FR)')
        elif item == 'reaction_progress_variable':
            ys.append(res_lambda)
            labels.append(r'$ \lambda $')
            long_labels.append(r'Reaction Progress Variable $ \lambda $')
        elif item == 'total_specific_energy':
            ys.append(res_E)
            labels.append(r'$ E^{total} $')
            long_labels.append(r'Total Specific Energy $ E^{total} $')
        elif item == 'internal_specific_energy':
            res_e_int = res_E - 0.5*res_v**2 + specific_heat_release*res_lambda
            ys.append(res_e_int)
            labels.append(r'$ e_{int} $')
            long_labels.append(r'Internal Specific Energy $ e_{int} $')
        elif item == 'Mach_number':
            res_M = np.abs(res_v) / res_c_s
            ys.append(res_M)
            labels.append(r'$ |M| $')
            long_labels.append(r'Mach Number $ |M| $')
        elif item == 'sound_speed':
            ys.append(res_c_s)
            labels.append(r'$ c_s $')
            long_labels.append(r'Local sound speed $ c_s $')
        elif item == 'reaction_rate':
            if k is None:
                k = preexponential_scaling_factor_k(
                    heat_capacity_ratio_gamma, specific_heat_release, activation_energy)
            res_om = arrhenius_reaction_rate(
                res_lambda,
                activation_energy,
                res_p,
                res_rho,
                k)
            ys.append(res_om)
            labels.append(r'$ \omega $')
            long_labels.append(r'Reaction Rate $ \omega $')

    if plot_reference:
        ref_p = ideal_reactive_gas_pressure_eos(
            ref_rho,
            ref_v,
            ref_E,
            ref_lambda,
            specific_heat_release,
            heat_capacity_ratio_gamma)
        ref_c_s = ideal_gas_sound_speed(
            ref_rho, ref_p, heat_capacity_ratio_gamma)

        # ys_ref = [ref_p, ref_rho, ref_v, ref_lambda, ref_E, np.abs(ref_v) / ref_c_s]
        ys_ref = []
        for item in what_to_plot:
            if item == 'pressure':
                ys_ref.append(ref_p)
            elif item == 'density':
                ys_ref.append(ref_rho)
            elif item == 'velocity':
                ys_ref.append(ref_v)
            elif item == 'reaction_progress_variable':
                ys_ref.append(ref_lambda)
            elif item == 'total_specific_energy':
                ys_ref.append(ref_E)
            elif item == 'internal_specific_energy':
                ref_e_int = ref_E - 0.5*ref_v**2 + specific_heat_release*ref_lambda
                ys_ref.append(ref_e_int)
            elif item == 'Mach_number':
                ref_M = np.abs(ref_v) / ref_c_s
                ys_ref.append(ref_M)
            elif item == 'sound_speed':
                ys_ref.append(ref_c_s)
            elif item == 'reaction_rate':
                if k is None:
                    k = preexponential_scaling_factor_k(
                        heat_capacity_ratio_gamma, specific_heat_release, activation_energy)
                ref_om = arrhenius_reaction_rate(
                    ref_lambda,
                    activation_energy,
                    ref_p,
                    ref_rho,
                    k)
                ys_ref.append(ref_om)
    # colours = [
    #     'tab:orange',
    #     'tab:green',
    #     'tab:red',
    #     'tab:blue',
    #     'tab:cyan',
    #     'magenta'
    # ]
    cmap = matplotlib.colormaps['plasma']
    # cmap = matplotlib.colormaps['inferno']
    # colours = cmap(np.linspace(0, 1, 6))
    colours = cmap(np.linspace(0.1, 0.8, len(what_to_plot)))
    colours1 = cmap(np.linspace(0.1-0.05, 0.8-0.05, len(what_to_plot)))

    # plt.rcParams['font.family'] = 'newPushkin'
    # plt.rcParams['font.serif'] = 'Cormorant Unicase'
    plt.rcParams['text.usetex'] = True
    # myfont = {'fontname': 'Cormorant Unicase'}
    myfont = {'fontname': 'Computer Modern Roman'}
    width, height = plt.figaspect(0.75)
    fig = plt.figure(figsize=(width, height), dpi=1200)
    if two_cols:
        gs = fig.add_gridspec(2, len(what_to_plot)//2)  # , hspace=0, wspace=0
        tpl1, tpl2 = gs.subplots(sharex='all')
        tpl = (*tpl1, *tpl2)
    else:
        gs = fig.add_gridspec(len(what_to_plot), 1, hspace=0)
        tpl = gs.subplots(sharex='all')
    if figname is not None:
        prename = figname
    else:
        prename = ''
    if plot_title:
        fig.suptitle(
            prename + ' Solution Profiles'
            # + ' with '
            # + fr'$ E_a = {int(activation_energy)},'
            # + fr'\quad q={int(specific_heat_release)},'
            # + fr'\quad \gamma={heat_capacity_ratio_gamma:.1f},'
            # + fr'\quad D_{{CJ}} \approx {detonation_speed:.1f} $'
            + fr' at Time $ t={solution_field["t"][0]:g} $',
            fontsize=12, **myfont)

    fig.patch.set_facecolor('none')
    cnt = 0
    for ax, func, colour, label, long_label in zip(
            tpl,
            ys,
            colours,
            labels,
            long_labels):
        if plot_reference:
            cl = colours1[cnt]
            func_ref = ys_ref[cnt]
            cnt += 1
        ax.set_facecolor('none')
        maxf = max(func)
        dif = abs(maxf - min(func))
        if dif / abs(maxf) < 1e-5:
            ax.set_ylim(maxf - 1e-5, maxf + 1e-5)
        ax.ticklabel_format(
            useOffset=useOffset,
            style='sci',
            # scilimits=scilimits,
            useMathText=True)
        if plot_grid:
            ax.grid(
                which='both', alpha=0.99, color='.8', linestyle='--', zorder=0)
        # ax.plot(xi, func, '.-', color=colour, label=label)
        # ax.plot(xi, func, color=colour, label=label, lw=0.8)
        # if plot_reference:
        #     ax.plot(
        #         xi, func_ref, color=cl, label='reference', lw=0.5,
        #         linestyle='dashed')
        ax.plot(xi, func, color=colour, label=label, lw=1.4)
        if plot_reference:
            # ax.plot(
            #     xi_ref, func_ref, color=cl, label='reference', lw=1.,
            #     linestyle='dashed')
            ax.plot(
                xi_ref, func_ref, color='black', label='reference', lw=1.,
                linestyle='dashed')
        # ax.grid(which='both', linestyle='-.')
        # ax.grid(which='both', linestyle='dotted')
        # ax.set_xlabel(r'Coordinate $ \xi $ in Lead Shock FR', **myfont)
        if not two_cols:
            ax.set_ylabel(label, **myfont)
            ax.yaxis.tick_right()
        for direction in (ax.xaxis, ax.yaxis):
            direction.label.set_size(8)
            direction.set_tick_params(labelsize=8)
        # ax.xaxis.set_major_locator(
        #     matplotlib.ticker.MultipleLocator(2))
        # ax.xaxis.set_minor_locator(
        #     matplotlib.ticker.MultipleLocator(1))
        ax.xaxis.set_major_locator(
            matplotlib.ticker.AutoLocator())
        ax.xaxis.set_minor_locator(
            matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(
            matplotlib.ticker.AutoMinorLocator())

        # ax.set_axisbelow(True)
        # ax.grid(which='both', alpha=0.99, color='.8')
        ax.tick_params(
            which='both',
            direction='in',
            right=True,
            # top=True,
            left=True,
            bottom=True)

        if add_subtitles:
            ax.add_artist(TickRedrawer())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='15%', pad=0)
            cax.get_xaxis().set_visible(False)
            cax.get_yaxis().set_visible(False)
            if use_black:
                cax.set_facecolor('k')
            if two_cols:
                cap_size = 8
            else:
                cap_size = 6.4

            if use_black:
                cax.text(
                    0.05,
                    0.45,
                    long_label,
                    size=cap_size,
                    color='white',
                    # bbox={'facecolor': 'black'},
                    ha='left',
                    va='center',
                    **myfont
                )
            else:
                cax.text(
                    0.05,
                    0.45,
                    long_label,
                    size=cap_size,
                    # color='white',
                    # bbox={'facecolor': 'black'},
                    ha='left',
                    va='center',
                    **myfont
                )

        if annotate_shock:
            if label == r'$ \lambda_0 $':
                ax.yaxis.set_minor_locator(
                    matplotlib.ticker.MultipleLocator(0.1))
                annot_max(xi, func, label, ax=ax, annot_min=True)
                ax.axhline(y=0.5, color='k', lw=0.5, ls='--')
                text = ax.text(
                    xi[0] + 10.*(xi[1]-xi[0])/(xi[-1]-xi[0]),
                    0.5,
                    "%.1f" % 0.5,
                    transform=ax.transData,
                    size=7,
                    rotation=0,
                    ha="right",
                    va="center",
                    **myfont
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(
                            linewidth=2, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
                ax.axvline(x=-1., color='k', lw=0.5, ls='--')
                text = ax.text(
                    -1,
                    0.95,
                    "%.0f" % -1.,
                    transform=ax.transData,
                    size=7,
                    rotation=90,
                    ha="center",
                    va="top",
                    **myfont
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(
                            linewidth=2, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
            else:
                annot_max(xi, func, label, ax=ax)

    # print(xi)
    if two_cols:
        tpl[-2].set_xlabel(coordinate_text, **myfont)
    else:
        tpl[-1].set_xlabel(coordinate_text, **myfont)

    fig.tight_layout()
    fig.align_ylabels()
    if savefig:
        plt.savefig(prename + 'reactive_euler_'
                        + fr'e_a={activation_energy:.2f}_'
                        + fr'q={specific_heat_release:.2f}_'
                        + fr'gamma={heat_capacity_ratio_gamma:.2f}'
                        + '.' + type,
                    # transparent=True,
                    format=type,
                    dpi='figure')
    plt.show()

    return res


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if (idx > 0 and (idx == len(array)
            or math.fabs(value - array[idx-1]) < math.fabs(
                value - array[idx]))):
        return idx - 1
    else:
        return idx


res6400 = np.load('./results/6400.npy')


# Mol = 28.96
# R = 287

np_arr = [100, 200, 400, 800, 1600]
# np_arr = [100, 200, 400, 800, 1600, 12800]
# np_arr = [100, 200, 400, 800, 1600, 12800]
# np_arr = [100, 200, 400, 800, 1600, 25600]
# np_arr = [101, 201, 401, 801, 1601, 25601]
# np_arr = [151, 301, 601, 1201, 2401, 38401]
n_lines = 0
x_arr = []
p_arr = []
rho_arr = []
T_arr = []
U_arr = []

for k in range(len(np_arr)):
    x_arr.append([])
    p_arr.append([])
    rho_arr.append([])
    T_arr.append([])
    U_arr.append([])
    with open(
            # 'postProcessing/singleGraph/0.4/line_T_p_rho_U.xy',
            f'results/{np_arr[k]}.xy',
            'r', newline='') as f:
        for line in f:
            n_lines += 1
            # xp, p, rho, T, U, _, _ = line.split(' ')
            xp, T, p, rho, U, _, _ = line.split(' ')
            x_arr[-1].append(np.float64(xp))
            p_arr[-1].append(np.float64(p))
            rho_arr[-1].append(np.float64(rho))
            T_arr[-1].append(np.float64(T))
            U_arr[-1].append(np.float64(U))

    # plt.plot(x_arr[-1], rho_arr[-1], label=f'$ N={np_arr[k]} $')


# rho_exact = np.array(rho_arr[-1])

rho_exact = np.array(res6400['solution'][0][..., 0][3:-3])
x_exact = np.array(res6400['coordinate_mesh'][0][3:-3])
np_arr.append(6400)
rho_arr.append(rho_exact)
U_arr.append(res6400['solution'][0][..., 1][3:-3] / rho_exact)
p_exact = ideal_eos_p(res6400['solution'][0][3:-3], 1.4, 50.)
p_arr.append(p_exact)
# T_arr.append(p_exact / rho_exact)
T_arr.append(res6400['solution'][0][..., 3][3:-3]
             / res6400['solution'][0][..., 0][3:-3])
x_arr.append(x_exact)

# plt.plot(x_arr[0], rho_arr[0], label=f'$ N={np_arr[0]} $')
# plt.plot(x_arr[0], abs(rho_arr[0] - rho_exact[::np_arr[-1]//(np_arr[0]-1)]))
plt.plot(x_arr[1], rho_arr[1], label=f'$ N={np_arr[1]} $')
# plt.plot(x_arr[1], abs(rho_arr[1] - rho_exact[::np_arr[-1]//(np_arr[1]-1)]))
plt.plot(x_arr[1], abs(rho_arr[1] - rho_exact[::np_arr[-1]//np_arr[1]]))
plt.grid()
plt.xlabel('$ x $')
plt.ylabel(r'$ \rho $')
plt.legend()
plt.show()


# dxes = 12. / (np.array(np_arr) - 1.)
dxes = 12. / np.array(np_arr)
l1_arr = []
l2_arr = []
linf_arr = []
rho_exact = np.array(rho_arr[-1])
U_exact = np.array(U_arr[-1])
T_exact = np.array(T_arr[-1])
p_exact = np.array(p_arr[-1])
x_exact = np.array(x_arr[-1])
for k in range(len(np_arr)):
    rho = np.array(rho_arr[k])
    U = np.array(U_arr[k])
    T = np.array(T_arr[k])
    p = np.array(p_arr[k])
    x = np.array(x_arr[k])
    l1 = 0.
    l2 = 0.
    linf = 0.
    for idx in range(len(rho)):
        idx_exact = find_nearest_index(x_exact, x[idx])
        abs_diff = abs(rho[idx] - rho_exact[idx_exact])
        # abs_diff = abs(rho[idx] - rho_exact[::np_arr[-1]//(np_arr[k]-1)][idx])
        # abs_diff = abs(rho[idx] - rho_exact[::np_arr[-1]//np_arr[k]][idx])
        # abs_diff = abs(U[idx] - U_exact[idx_exact])
        # abs_diff = abs(T[idx] - T_exact[idx_exact])
        # abs_diff = abs(p[idx] - p_exact[idx_exact])
        linf = max(abs_diff, linf)
        l1 += abs_diff ** 1
        l2 += abs_diff ** 2
    # l1 *= 12. / np_arr[k]
    # l1 *= 12. / (np_arr[k] - 1.)
    l1 *= dxes[k]
    l1 **= 1.
    # l2 *= 12. / np_arr[k0]
    # l2 *= 12. / (np_arr[k] - 1.)
    l2 *= dxes[k]
    l2 **= 0.5
    if len(np_arr)-1 > k > 0:
        factor = 1. / (np.log(dxes[k-1]) - np.log(dxes[k]))
        print(k, dxes[k], 'done:',
              'l_inf', linf, factor * (np.log(linf_arr[-1]) - np.log(linf)),
              '|', 'l1', l1, factor * (np.log(l1_arr[-1]) - np.log(l1)),
              '|', 'l2', l2, factor * (np.log(l2_arr[-1]) - np.log(l2)))
    else:
        print(k, dxes[k], 'done:',
              'l_inf', linf, 'n/a',
              '|', 'l1', l1, 'n/a',
              '|', 'l2', l2, 'n/a')
    l1_arr.append(l1)
    l2_arr.append(l2)
    linf_arr.append(linf)

l1_arr = np.array(l1_arr)[:-1]
l2_arr = np.array(l2_arr)[:-1]
linf_arr = np.array(linf_arr)[:-1]
dxes = dxes[:-1]
# print(dxes)
# print([np.mean(np.diff(x_arr[k])) for k in range(len(x_arr))])
fig, ax = plt.subplots(1)
ax.plot(dxes, l1_arr, '.-', label='$ L_1 $')
ax.plot(dxes, l2_arr, '.-', label='$ L_2 $')
ax.plot(dxes, linf_arr, '.-', label=r'$ L_\infty $')
ax.plot(dxes, (linf_arr[0]/dxes[0]**2) * dxes**2,
        '--', color='red', label='$ O(dx^2) $')
# ax.plot(dxes, (l1_arr[0]/dxes[0]**2) * dxes**2,
#         '--', color='red', label='$ O(dx^2) $')
# ax.plot(dxes, (l2_arr[0]/dxes[0]**1.7) * dxes**1.7,
#         '-.', color='red', label=r'$ O(dx^{1.7}) $')
# ax.plot(dxes, (l2_arr[0]/dxes[0]**1) * dxes**1,
#         linestyle=(0, (1, 10)), color='red', label='$ O(dx^1) $')
# ax.plot(dxes, (linf_arr[0]/dxes[0]**1) * dxes**1,
#         linestyle=(0, (1, 10)), color='red', label='$ O(dx^1) $')
ax.plot(dxes, (linf_arr[0]/dxes[0]**1) * dxes**1,
        '-.', color='red', label='$ O(dx^1) $')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xticks(dxes)
labels = dxes
ax.set_xticklabels(labels)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0)
ax.set_xlabel(r'Mesh size, $ dx $')
ax.set_ylabel(r'Error, $ \varepsilon $')
ax.grid(axis='both', which='major')
ax.grid(axis='both', which='minor', linestyle='--')

plt.grid()
plt.legend()
plt.savefig('convergence.svg', dpi=1200)
plt.show()


bound_size = 0
n_pts = len(x_arr[-2])
field_dtype = np.dtype([
    ('coordinate_mesh', np.float64, (n_pts + 2*bound_size,)),
    ('initial_data', np.float64, (n_pts + 2*bound_size, 4)),
    ('solution', np.float64, (n_pts + 2*bound_size, 4)),
    ('reference_solution', np.float64, (n_pts + 2*bound_size, 4)),
    ('t', np.float64),
    ('number_of_points', np.float64)
], align=True)

res_field = np.empty(1, dtype=field_dtype)
res_field['t'] = 3.0
res_field['number_of_points'] = n_pts
res_field['solution'][0][..., 0] = rho_arr[-2]
res_field['solution'][0][..., 1] = np.array(rho_arr[-2]) * np.array(U_arr[-2])
res_field['solution'][0][..., 2] = (np.array(rho_arr[-2]) * np.array(U_arr[-2])**2 * 0.5
                                    + np.array(p_arr[-2]) / (1.4 - 1.)
                                    - 50. * np.array(rho_arr[-2]) * np.array(T_arr[-2]))
res_field['solution'][0][..., 3] = np.array(rho_arr[-2]) * np.array(T_arr[-2])
res_field['initial_data'] = res_field['solution']
res_field['coordinate_mesh'] = np.array(x_arr[-2])

plot_reactive_euler_solution_profiles(
    res_field, {
        'heat_capacity_ratio': 1.4,
        'specific_heat_release': 50.,
        'activation_energy': 10.,
        'frame_of_reference_speed': 0.
    },
    plot_reference=True,
    savefig=True,
    type='png',
    two_cols=True,
    figname='Detonation Initiation (Xu et al., 1997)',
    what_to_plot=[
        'pressure',
        'density',
        'velocity',
        'internal_specific_energy',
        # 'total_specific_energy',
        # 'Mach_number',
        # 'sound_speed',
        'reaction_progress_variable',
        'reaction_rate'
    ],
    reference=res6400, k=7.);


plt.plot(x_exact, rho_exact, '.-')
plt.plot(x_arr[-2], rho_arr[-2], '.-')
plt.show()
