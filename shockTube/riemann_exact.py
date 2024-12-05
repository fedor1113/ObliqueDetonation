#!/usr/bin/env python3

import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp


def ideal_gas_eos_primitive(primitives,
                            gamma: float = 1.4) -> float:
    # rho, u, p = primitives
    rho = primitives[..., 0]
    u = primitives[..., 1]
    p = primitives[..., 2]
    rhoeint = p / (gamma - 1.)

    # rhoE
    return rhoeint + 0.5*rho*(u**2)


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


def conservative_from_primitive(flow_var_vec, eoses):
    # n_phases = len(eoses)
    res = np.zeros_like(flow_var_vec)
    for i, eos in enumerate(eoses):
        start = i*4
        end = start + 4
        current_flow_var_vec = flow_var_vec[..., start:end]
        # total_specific_energy = eos(flow_var_vec[i*4:(i+1)*4])
        total_specific_energy = eos(current_flow_var_vec[..., :-1]) / current_flow_var_vec[..., 0]
        res[..., start+0] = current_flow_var_vec[..., 0] * 1
        res[..., start+1] = current_flow_var_vec[..., 0] * current_flow_var_vec[..., 1]
        res[..., start+2] = current_flow_var_vec[..., 0] * total_specific_energy
        res[..., start+3] = current_flow_var_vec[..., -1]  # this remains as non-conserved

    return res


def toro_exact_sol(
        left_state_primitive,
        right_state_primitive,
        sol_type,
        shock_position,
        heat_capacity_ratio,
        number_of_points,
        left_bound,
        right_bound,
        t_max,
        boundary_size: int = 1,
        left_biased: bool = False):
    # TO DO:
    # - automatic wave structure finder,
    # - better initial approx for p_star

    if left_biased:
        less = lambda x, y: x <= y
    else:
        less = lambda x, y: x < y

    bound_size = boundary_size
    gamma = heat_capacity_ratio
    x0 = shock_position
    t = t_max

    if len(left_state_primitive) < 4:
        rl, ul, pl = left_state_primitive
        rr, ur, pr = right_state_primitive
        ll = 0.
        lr = 0.
    else:
        rl, ul, pl, ll = left_state_primitive
        rr, ur, pr, lr = right_state_primitive

    gam_m = (gamma - 1.)
    gam_p = (gamma + 1.)
    al = (gamma * pl / rl) ** 0.5
    ar = (gamma * pr / rr) ** 0.5


    AL = 2. / gam_p / rl
    BL = (gam_m / gam_p) * pl
    AR = 2. / gam_p / rr
    BR = (gam_m / gam_p) * pr


    WLfan_v = lambda xc: (2. / gam_p) * (al + 0.5 * gam_m * ul + xc / t)
    WRfan_v = lambda xc: (2. / gam_p) * (-ar + 0.5 * gam_m * ur + xc / t)
    left_RW_head_speed = ul - al
    right_RW_head_speed = ur + ar

    WLfan_r = lambda xc: rl * ((2. / gam_p)
                               + (gam_m/gam_p/al)*(ul - xc / t))**(2/gam_m)
    WRfan_r = lambda xc: rr * ((2. / gam_p)
                               - (gam_m/gam_p/ar)*(ur - xc / t))**(2/gam_m)

    WLfan_p = lambda xc: pl * ((2. / gam_p)
                               + (gam_m/gam_p/al)*(ul - xc / t))**(2*gamma/gam_m)
    WRfan_p = lambda xc: pr * ((2. / gam_p)
                               - (gam_m/gam_p/ar)*(ur - xc / t))**(2*gamma/gam_m)

    if sol_type == 'RW-CD-SW':
        fl = lambda p: 2 * al / gam_m * ((p/pl) ** (gam_m / (2. * gamma)) - 1)
        fr = lambda p: (p - pr) * (AR / (p + BR))**0.5
    elif sol_type == 'RW-CD-RW':
        fl = lambda p: 2 * al / gam_m * ((p/pl) ** (gam_m / (2. * gamma)) - 1)
        fr = lambda p: 2 * ar / gam_m * ((p/pr) ** (gam_m / (2. * gamma)) - 1)
    elif sol_type == 'SW-CD-RW':
        fl = lambda p: (p - pl) * (AL / (p + BL))**0.5
        fr = lambda p: 2 * ar / gam_m * ((p/pr) ** (gam_m / (2. * gamma)) - 1)
    elif sol_type == 'SW-CD-SW':
        fl = lambda p: (p - pl) * (AL / (p + BL))**0.5
        fr = lambda p: (p - pr) * (AR / (p + BR))**0.5

    func = lambda p: fl(p) + fr(p) + ur - ul
    # p_star_linearized_guess = max(
    #     10**(-9),
    #     0.5*(pl + pr) - 0.125*(ur - ul)*(rl + rr)*(al + ar))
    p_two_rarefaction_guess = (
        (al + ar - 0.5*gam_m*(ur - ul))
        / (al / pl**(gam_m/gamma/2.)
           + ar / pr**(gam_m/gamma/2.))
    ) ** (2.*gamma/gam_m)
    # p_star = scipy.optimize.fsolve(func, 0.3)[0]
    # p_star = scipy.optimize.fsolve(func, 0.0018)[0]
    # p_star = scipy.optimize.fsolve(func, 460)[0]
    ## p_star = scp.optimize.fsolve(func, 46)[0]
    # p_star = scipy.optimize.fsolve(func, 1691)[0]
    p_star = scp.optimize.fsolve(
        func, p_two_rarefaction_guess, xtol=1e-14, maxfev=100000)[0]

    if not np.isclose(func(p_star), 0., atol=2e-15):
        g_k = lambda x: (x[0] / (x[1] + x[2])) ** 0.5

        p_star_linearized_guess = max(
            2. * (10**(-15)),
            0.5*(pl + pr) - 0.125*(ur - ul)*(rl + rr)*(al + ar))
        p_star0 = p_star_linearized_guess

        gl = g_k((AL, p_star0, BL))
        gr = g_k((AR, p_star0, BR))
        p_two_shock_guess = max(
            2. * (10**(-15)),
            (gl*pl + gr*pr - (ur - ul)) / (gl + gr))

        p_star1 = scp.optimize.fsolve(
            func, p_two_shock_guess, xtol=1e-14, maxfev=100000)[0]
        if np.abs(func(p_star)) > np.abs(func(p_star1)):
            p_star = p_star1

    print(np.isclose(func(p_star), 0., atol=2e-15), func(p_star))

    u_star = 0.5 * (ul + ur) + 0.5 * (fr(p_star) - fl(p_star))

    Ql = ((p_star + BL) / AL) ** 0.5
    Qr = ((p_star + BR) / AR) ** 0.5
    right_SW_speed = ur + Qr / rr
    left_SW_speed = ul - Ql / rl
    left_SW_position = x0 + left_SW_speed * t
    right_SW_position = x0 + right_SW_speed * t

    left_RW_head_position = x0 + left_RW_head_speed * t
    right_RW_head_position = x0 + right_RW_head_speed * t
    a_star_left = al * (p_star / pl) ** (gam_m / (2. * gamma))
    a_star_right = ar * (p_star / pr) ** (gam_m / (2. * gamma))
    left_RW_tail_speed = u_star - a_star_left
    left_RW_tail_position = x0 + left_RW_tail_speed * t
    right_RW_tail_speed = u_star + a_star_right
    right_RW_tail_position = x0 + right_RW_tail_speed * t

    CW_position = x0 + u_star*t
    # r_star_left_RW = (a_star_left**2 / p_star / gamma)**(-1)
    r_star_left_RW = rl * (p_star / pl)**(1./gamma)
    r_star_left_SW = (rl
                      * (p_star/pl + gam_m/gam_p)
                      / ((gam_m/gam_p * p_star/pl) + 1.))

    # r_star_right_RW = (a_star_right**2 / p_star / gamma)**(-1)
    r_star_right_RW = rr * (p_star / pr)**(1./gamma)
    r_star_right_SW = (rr
                       * (p_star/pr + gam_m/gam_p)
                       / ((gam_m/gam_p * p_star/pr) + 1.))


    def sol_at(coord, sol_type='RW-CD-RW'):
        if sol_type == 'RW-CD-SW':
            if less(coord, left_RW_head_position):
                return rl, ul, pl, ll
            if less(coord, left_RW_tail_position):
                return WLfan_r(coord-x0), WLfan_v(coord-x0), WLfan_p(coord-x0), WLfan_r(coord-x0)*ll
            if less(coord, right_SW_position):
                if less(coord, CW_position):
                    return r_star_left_RW, u_star, p_star, r_star_left_RW*ll
                else:
                    return r_star_right_SW, u_star, p_star, r_star_right_RW*lr
            return rr, ur, pr, lr
        elif sol_type == 'RW-CD-RW':
            if less(coord, left_RW_head_position):
                return rl, ul, pl, ll
            if less(coord, left_RW_tail_position):
                return WLfan_r(coord-x0), WLfan_v(coord-x0), WLfan_p(coord-x0), WLfan_r(coord-x0)*ll
            if less(coord, right_RW_tail_position):
                if less(coord, CW_position):
                    return r_star_left_RW, u_star, p_star, r_star_left_RW*ll
                else:
                    return r_star_right_RW, u_star, p_star, r_star_right_RW*lr
            if less(coord, right_RW_head_position):
                return WRfan_r(coord-x0), WRfan_v(coord-x0), WRfan_p(coord-x0), WRfan_r(coord-x0)*lr
            return rr, ur, pr, lr
        elif sol_type == 'SW-CD-RW':
            if less(coord, left_SW_position):
                return rl, ul, pl, ll
            if less(coord, right_RW_tail_position):
                if less(coord, CW_position):
                    return r_star_left_SW, u_star, p_star, r_star_left_SW*ll
                else:
                    return r_star_right_RW, u_star, p_star, r_star_right_RW*lr
            if less(coord, right_RW_head_position):
                return WRfan_r(coord-x0), WRfan_v(coord-x0), WRfan_p(coord-x0), WRfan_r(coord-x0)*lr
            return rr, ur, pr, lr
        elif sol_type == 'SW-CD-SW':
            if less(coord, left_SW_position):
                return rl, ul, pl, ll
            if less(coord, right_SW_position):
                if less(coord, CW_position):
                    return r_star_left_SW, u_star, p_star, r_star_left_SW*ll
                else:
                    return r_star_right_SW, u_star, p_star, r_star_right_SW*lr
            return rr, ur, pr, lr

    n_pts = number_of_points
    a = left_bound
    b = right_bound
    dx = (b - a) / (n_pts - 1)
    x = np.zeros(n_pts + 2*bound_size)
    # x[3:-3] = np.linspace(-1., 1., n_pts)
    x[bound_size:-bound_size] = np.asarray(
        [a + k*dx for k in range(n_pts)])
    x[0:bound_size] = np.asarray(
        [a - k*dx for k in range(1, bound_size+1)])[::-1]
    x[-bound_size:] = np.asarray(
        [b + k*dx for k in range(1, bound_size+1)])

    res = np.array(
        [conservative_from_primitive(
            # np.concatenate([np.array(sol_at(c, sol_type)), np.array([0.])]),
            np.array(sol_at(c, sol_type)),
            (ideal_gas_eos_primitive,))
         for c in x])

    return x, res


number_of_points = 10001
toro1 = {
    'left_state_primitive': (1., 0.75, 1., 0.),
    'right_state_primitive': (0.125, 0., 0.1, 0.),
    'shock_position': 0.3,
    'heat_capacity_ratio': 1.4,
    # 'specific_heat_release': 0.,
    # 'activation_energy': np.inf,
    'number_of_points': number_of_points,
    'left_bound': 0.,
    'right_bound': 1.,
    't_max': 0.2,
    # 'cfl': cfl,
    'sol_type': 'RW-CD-SW'
}

x, res = toro_exact_sol(**toro1)


# Mol = 28.96
# R = 287

number_of_points = 10001
n_lines = 0
x_arr = []
p_arr = []
rho_arr = []
T_arr = []
U_arr = []

with open(
        'postProcessing/singleGraph/0.007/line_T_p_rho_U.xy',
        'r', newline='') as f:
    for line in f:
        n_lines += 1
        # xp, p, rho, T, U, _, _ = line.split(' ')
        xp, T, p, rho, U, _, _ = line.split(' ')
        x_arr.append(np.float64(xp))
        p_arr.append(np.float64(p))
        rho_arr.append(np.float64(rho))
        T_arr.append(np.float64(T))
        U_arr.append(np.float64(U))


Mol = 28.96
# R = 287
R = 8.314472
# R = 8314.
# R_g = R / Mol  # = 9.9102209944751381215469613259669
R_g = R / Mol * 10**3
# R_g = 287.
toro_of = {
    'left_state_primitive': (100000./(R_g*348.432), 0., 100000., 0.),
    'right_state_primitive': (10000./(R_g*278.746), 0., 10000., 0.),
    'shock_position': 0.,
    'heat_capacity_ratio': 1.4,
    # 'specific_heat_release': 0.,
    # 'activation_energy': np.inf,
    'number_of_points': number_of_points,
    'left_bound': -5.,
    'right_bound': 5.,
    't_max': 0.007,
    # 'cfl': cfl,
    'sol_type': 'RW-CD-SW'
}
# toro_of['number_of_points'] = n_lines - 1
toro_of['number_of_points'] = n_lines
x, res = toro_exact_sol(**toro_of)
# toro_of = {
#     'left_state_primitive': (1., 0., 1./(R_g*0.00348432), 0.),
#     'right_state_primitive': (.1, 0., .1/(R_g*0.00278746), 0.),
#     'shock_position': 0.,
#     'heat_capacity_ratio': 1.4,
#     # 'specific_heat_release': 0.,
#     # 'activation_energy': np.inf,
#     'number_of_points': number_of_points,
#     'left_bound': 0.,
#     'right_bound': 1.,
#     # 't_max': 0.007,
#     't_max': 0.1,
#     # 'cfl': cfl,
#     'sol_type': 'RW-CD-SW'
# }
# x, res = toro_exact_sol(**toro_of)
# plt.plot(x, ideal_gas_eos_p(res))
# plt.grid()
# plt.show()

p_exact = []
rho_exact = []
v_exact = []
T_exact = []
with open('sod.csv', 'w', newline='') as csvfile:
    fieldnames = ['x', 'p_exact', 'rho_exact', 'v_exact', 'T_exact']
    # fieldnames = ['x coord', 'scalar']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writerow({f: f for f in fieldnames})
    for i in range(1, len(x)-1):
        # writer.writerow({'x coord': x[i], 'scalar': ideal_gas_eos_p(res)[i]})
        p = ideal_gas_eos_p(res)[i]
        writer.writerow({'x': x[i],  #  - x[0]
                         'p_exact': p,
                         'rho_exact': res[i, 0],
                         'T_exact': p/res[i, 0]/R_g,
                         'v_exact': res[i, 1]/res[i, 0]})
        p_exact.append(p)
        rho_exact.append(res[i, 0])
        v_exact.append(res[i, 1]/res[i, 0])
        T_exact.append(p/res[i, 0]/R_g)

l2add = [0., 0., 0., 0.]
l2 = [0., 0., 0., 0.]
l1 = [0., 0., 0., 0.]
linf = [0., 0., 0., 0.]
# print(n_lines)
print(f'{len(x[1:-1])=}')
for k in range(n_lines):
    # print(x[k+1], x_arr[k], p_exact[k], p_arr[k])
    l2add[0] = np.abs(p_exact[k] - p_arr[k])
    l2add[1] = np.abs(rho_exact[k] - rho_arr[k])
    l2add[2] = np.abs(T_exact[k] - T_arr[k])
    l2add[3] = np.abs(v_exact[k] - U_arr[k])
    for j in range(4):
        l1[j] += abs(l2add[j])

        l2add[j] **= 2
        l2[j] += abs(l2add[j])

dx = (x[-1] - x[0]) / (n_lines - 1)
for k in range(4):
    l1[k] *= dx

    l2[k] *= dx
    l2[k] **= 0.5

linf[0] = np.max(np.abs(np.array(p_arr) - np.array(p_exact)))
linf[1] = np.max(np.abs(np.array(rho_arr) - np.array(rho_exact)))
linf[2] = np.max(np.abs(np.array(T_arr) - np.array(T_exact)))
linf[3] = np.max(np.abs(np.array(U_arr) - np.array(v_exact)))

plt.plot(x_arr, p_arr, '.-')
plt.plot(x_arr, p_exact)
plt.grid()
plt.show()
plt.plot(x_arr, np.array(p_arr) - np.array(p_exact))
plt.grid()
plt.show()
print(f'{dx=}')
print('coordinate max error',
      np.max(np.abs(np.array(x_arr) - np.array(x[1:-1]))))
print('p, rho, T, U L1 errors')
print(l1)
print('p, rho, T, U L2 errors')
print(l2)
print('p, rho, T, U Linf errors')
print(linf)

# print()
# print(np.sum(np.abs(np.array(p_arr) - np.array(p_exact))))
# print(np.sum(np.abs(np.array(p_arr) - np.array(p_exact)))*dx)
# print((np.sum(np.abs(np.array(p_arr) - np.array(p_exact))**2)*dx)**0.5)
# print(np.max(np.abs(np.array(p_arr) - np.array(p_exact))))
# print(np.sum(np.abs(np.array(p_arr) - np.array(p_exact)))/n_lines)
# print((np.sum(np.abs(np.array(p_arr) - np.array(p_exact))**2)/n_lines)**0.5)
# print(np.max(np.abs(np.array(p_arr) - np.array(p_exact))))
# print(dx)
# print(1/n_lines)
