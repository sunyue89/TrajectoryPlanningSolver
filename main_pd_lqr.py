def build_lqr():
    n = problem.num_states()
    m = problem.num_controls()
    N = problem.num_stages()
    for i in range(N):
    cost_grad = problem.cost(i)
    .gradient()
    .eval(x[:,i], u[:,i])
    dyn_grad = problem.dynamics(i)

    .gradient()
    .eval(x[:,i], u[:,i])
    A_i = dyn_grad.T[:,:n]
    B_i = dyn_grad.T[:,n:]
    if i == 0:
    d_i = problem.initial_state()
    - x[:,0]
    else:
    y = problem.dynamics(i-1)
    .eval(x[:,i-1], u[:,i-1])
    d_i = y - x[:,i]
    l_i = cost_grad +
    dyn_grad * costates[:,i+1]
    - costates[:,i]
    cost_hess = problem.cost(i)
    .hessian()
    .eval(x[:,i], u[:,i])
    dyn_hess = problem.dynamics(i)
    .hessians()
    .eval(x[:,i], u[:,i])
    Q_i = cost_hess
    + dyn_hess * costates[:,i+1]
    l_N = problem.cost(N).gradient()
    .eval(x[:,N]) - costates[:,N]
    Q_N = problem.cost(N).hessian()
    .eval(x[:,N])
    return LQR(A_*, B_*, d_*, Q_*, l_*)

def primal_dual_ilqr():
    x = states_ws
    u = controls_ws
    v = costates_ws
    for it in range(kMaxIterations):
    lqr = build_lqr(problem, x, u)
    dx, du, P, p = lqr.Solve()
    dv = lqr_dual_solve(P, p, dx)
    if dx == 0 and du == 0:
    if dv == 0:
    return x, u, v
    else:
    v += dv
    continue
    rho = merit_rho(lqr.d(), dv)
    alpha = line_search(
    problem, lqr, x, u,
    v, dx, du, dv, rho)
    x += alpha * dx
    u += alpha * du
    v += alpha * dv
    return x, u, v