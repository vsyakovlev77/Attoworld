import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import attoworld as aw
    import matplotlib.pyplot as plt
    return aw, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""## Check convergence of numerical derivatives""")
    return


@app.cell
def _(aw, np, plt):
    def convergence_check(order, N_pts):
        x = np.linspace(0.0, 2*np.pi, N_pts+1)[0:-1]
        dx = x[1]-x[0]
        y = np.cos(x)
        y_derivative_analytic = -np.sin(x)
        y_derivative = aw.numeric.uniform_derivative(y,1,order,boundary='periodic')/dx
        return np.max(np.abs(y_derivative - y_derivative_analytic))

    N_pts_range = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    order_range = range(1,6)

    convergence_data = np.array([[convergence_check(_order, _n) for _n in N_pts_range] for _order in order_range])

    lines = plt.loglog(N_pts_range,convergence_data.T)
    for order, line in zip(order_range, lines):
        line.set_label(f"{order}")
    plt.xlabel("Number of grid points")
    plt.ylabel("Mean error")
    plt.ylim(1e-15, 0.1)
    plt.legend()
    aw.plot.showmo()
    return (N_pts_range,)


@app.cell
def _(mo):
    mo.md(r"""## Check interpolate()""")
    return


@app.cell
def _(aw, np, plt):
    def plot_interpolate_test():
        x = np.real(np.linspace(0.0,17.0,24))
        x_fine = np.real(np.linspace(0.0,17.0,1024))
        y = np.sin(x**2/10)
        y_fine = np.sin(x_fine**2/10)
        x2 = np.linspace(0.0,19.0,60)
        plt.plot(x,y,'o')
        plt.plot(x_fine,y_fine)
        plt.plot(x2,aw.numeric.interpolate(x2, x,y, 3, extrapolate=False),'x')

    plot_interpolate_test()
    aw.plot.showmo()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Check FWHM error
    Note that the aw.numeric.fwhm() function has a much smaller error than the implementation I've seen elsewhere (which only reduces in error linearly with step size).
    """
    )
    return


@app.cell
def _(N_pts_range, aw, np, plt):
    def plot_fwhm_convergence():
        true_fwhm = 2*np.sqrt(2.0 * np.log(2.0))
        fwhm = []
        fwhm_th = []
        dx_array = []
        def old_fwhm(y, x):
            i_max = np.argmax(y)
            hm = y[i_max] / 2
            return (np.max(x[np.argwhere(y >= hm)]) - np.min(x[np.argwhere(y >= hm)]))
        
        for i in range(len(N_pts_range)):
            x = np.linspace(-5, 5, N_pts_range[i])
            dx = x[1]-x[0]
            y = np.exp(-x**2/2)
            fwhm.append(aw.attoworld_rs.fwhm(y, dx))
            fwhm_th.append(old_fwhm(y,x))
            dx_array.append(dx)
        plt.loglog(N_pts_range, dx_array, label="time step")
        plt.loglog(N_pts_range, np.abs(fwhm-true_fwhm), label="FWHM in attoworld.numeric")
        plt.loglog(N_pts_range, np.abs(fwhm_th-true_fwhm), label="index-based FWHM")
        plt.xlabel("# of points")
        plt.ylabel("Error")
        plt.legend()


    plot_fwhm_convergence()
    aw.plot.showmo()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
