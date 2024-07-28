import numpy as np
import pyshtools as pysh
import pickle
import scipy
from matplotlib import pyplot as plt
import pygmt
import ctplanet as ctp
import datetime
import time

def plotTopo(hlm,limits,central_longitude,title,show,labelling=None):
    hlm.coeffs[0, 0, 0] = 0

    topo = hlm.expand() / 1000

    if show==True:
        fig2 = topo.plotgmt(projection='robinson',
                            central_longitude=central_longitude,
                            cmap='roma',
                            cmap_limits=limits,
                            tick_interval=None,
                            cmap_reverse=True,
                            colorbar='bottom',
                            cb_label='Elevation',
                            cb_ylabel='km',
                            cb_triangles='both',
                            cb_tick_interval=1,
                            grid=True,
                            shading=True,
                            title=title)

        if labelling == True:
            fig2.text(text="A  p  h  r  o  d  i  t  e    T  e  r  r  a", x=130, y=-1.5, angle=5,
                      font="12p,Palatino-BoldItalic,black")
            fig2.text(text="I s h t a r  T e r r a", x=1, y=70, angle=0, font="12p,Palatino-BoldItalic,black")
            fig2.text(text="Atlanta", x=165, y=75, angle=0, font="9p,Palatino-BoldItalic,black")
            fig2.text(text="Planitia", x=155, y=65, angle=0, font="9p,Palatino-BoldItalic,black")
            fig2.text(text="Aino Planitia", x=120, y=-55, angle=-17, font="12p,Palatino-BoldItalic,black")
            fig2.text(text="Lavinia", x=-10, y=-40, angle=0, font="9p,Palatino-BoldItalic,black")
            fig2.text(text="Planitia", x=-10, y=-50, angle=0, font="9p,Palatino-BoldItalic,black")
            fig2.text(text="Sedna Planitia", x=-10, y=40, angle=-30, font="9p,Palatino-BoldItalic,black")
            fig2.text(text="Guinevere Planitia", x=-50, y=35, angle=-55, font="9p,Palatino-BoldItalic,black")
        fig2.show()
    return topo





def plotFreeAir(clm,lmax,a,f,limits,central_longitude,title,discrete,show,step=None):
    grav_all = clm.expand(lmax=lmax, a=a, f=f) #Gravity Field
    fa_grid = grav_all.total*10e5 # Convert the total anomaly in mGal from m/s^2
    if discrete == True:
        fa_grid_discrete = np.zeros((fa_grid.data.shape[0], fa_grid.data.shape[1]))

        for j in range(0, fa_grid.data.shape[0]):
            for k in range(0, fa_grid.data.shape[1]):
                for i in range (0,limits[0],step):
                    if fa_grid.data[j][k] < limits[0]:
                        fa_grid_discrete[j][k] = limits[0]
                    if fa_grid.data[j][k] > limits[0]-step*limits[0]:
                        fa_grid_discrete[j][k] = limits[0]-step*limits[0]
        fa_grid = pysh.SHGrid.from_array(fa_grid_discrete)
    if show==True:
        fig3 = fa_grid.plotgmt(projection='robinson',
                               central_longitude=central_longitude,
                               cmap='roma',
                               cmap_limits=limits,
                               tick_interval=None,
                               cmap_reverse=True,
                               colorbar='bottom',
                               cb_label='Gravity acceleration',
                               cb_ylabel='mgal',
                               grid=True,
                               title=title,
                               shading=False)

        fig3.show()

    return fa_grid




def plotBouguer(clm,hlm,lmax,crust_density,mantle_density,a,f,limits,central_longitude,title,type,discrete,show,step=None,nmax=None,R=None,M=None,storage=None,savename=None):
    hlm = pysh.datasets.Venus.VenusTopo719(lmax=lmax)
    if type == 'SHTools':
        bc = pysh.SHGravCoeffs.from_shape(shape=hlm,
                                          rho=crust_density,
                                          gm=clm.gm,
                                          lmax=lmax)

        bc = bc.change_ref(r0=clm.r0)


        ell = pysh.SHGrid.from_ellipsoid(lmax=bc.lmax,
                                         a=a,
                                         b=a-a*f)

        elm = pysh.SHGravCoeffs.from_shape(shape=ell,
                                           rho=crust_density,
                                           gm=clm.gm)

        elm = elm.change_ref(r0=clm.r0)

        ba = clm - bc + elm

        ba_grid = ba.expand(lmax=lmax,
                            a=a,
                            f=f)

        if discrete == True:

            ba_grid_discrete = np.zeros((ba_grid.total.data.shape[0], ba_grid.total.data.shape[1]))

            for j in range(0, ba_grid.data.shape[0]):
                for k in range(0, ba_grid.data.shape[1]):
                    for i in range(0, limits[0], step):
                        if ba_grid.data[j][k] < limits[0]:
                            ba_grid_discrete[j][k] = limits[0]
                        if ba_grid.data[j][k] > limits[0] - step * limits[0]:
                            ba_grid_discrete[j][k] = limits[0] - step * limits[0]
            ba_grid = pysh.SHGrid.from_array(ba_grid_discrete)

        if show==True:
            fig4 = (ba_grid.total*1e5).plotgmt(projection='robinson',
                                         central_longitude=central_longitude,
                                         cmap='roma',
                                         cmap_limits=limits,
                                         tick_interval=None,
                                         cmap_reverse=True,
                                         colorbar='bottom',
                                         cb_label='Gravity acceleration',
                                         cb_ylabel='mgal',
                                         title=title,
                                         grid=True,
                                         shading=False)

            fig4.show()
        return ba_grid, bc

    if type == 'FirstOrder':
        qlmi = np.zeros((2, lmax+1, lmax+1))  # Coefficients qlm are the gravity coefficients related to bouguer correction
        for l in range(1, lmax+1):
            for m in range(0, lmax+1):
                qlmi[0][l][m] = (4*np.pi*R**2*crust_density*hlm.coeffs[0][l][m])/(M*(2*l+1))
                qlmi[1][l][m] = (4*np.pi*R**2*crust_density*hlm.coeffs[1][l][m])/(M*(2*l+1))

        qlm = pysh.SHGravCoeffs.from_array(qlmi, gm=clm.gm, r0=clm.r0)
        qlm = qlm.change_ref(r0=clm.r0)

        ell1 = pysh.SHGrid.from_ellipsoid(lmax=qlm.lmax,
                                         a=a,
                                         b=a-a*f)

        elm1 = pysh.SHGravCoeffs.from_shape(shape=ell1,
                                           rho=crust_density,
                                           gm=clm.gm)

        elm1 = elm1.change_ref(r0=clm.r0)

        ba_fo = clm - qlm + elm1  # Gravity coefficients of bouguer anomaly

        # Computing Bouguer anomaly map

        ba_grid_fo = ba_fo.expand(a=a, f=f)  # Expansion into a grid

        if show==True:
            fig5 = (ba_grid_fo.total*1e5).plotgmt(projection='robinson',
                                                     central_longitude=central_longitude,
                                                     cmap='roma',
                                                     cmap_limits=limits,
                                                     tick_interval=None,
                                                     cmap_reverse=True,
                                                     colorbar='bottom',
                                                     cb_label='Gravity acceleration',
                                                     cb_ylabel='mgal',
                                                     title=title,
                                                     grid=True,
                                                     shading=False)

            fig5.show()
        return ba_grid_fo, qlm

    if type == 'NExpansion':

        if storage == False:
            topo = hlm.expand(lmax=lmax) - R

            prod = 1.0
            sum0 = 0.0
            sum1 = 0.0

            hlm_list = []
            for n in range(1, nmax + 1):
                topo_n = topo ** n
                hlm_n = topo_n.expand()
                hlm_list.append(hlm_n)

            klmi = np.zeros((2, lmax+1, lmax+1))
            for l in range(1, lmax+1):
                for m in range(0, lmax+1):
                    for n in range(1, nmax+1):
                        for j in range(1, n+1):
                            prod *= l+4-j
                        hlm_n = hlm_list[n-1]
                        sum0 += (hlm_n.coeffs[0][l][m])/(R**n*scipy.special.factorial(n)*(l+3))*prod
                        sum1 += (hlm_n.coeffs[1][l][m])/(R**n*scipy.special.factorial(n)*(l+3))*prod
                        prod = 1
                    klmi[0][l][m] = (4*np.pi*R**3*crust_density)/(M*(2*l+1))*sum0
                    klmi[1][l][m] = (4*np.pi*R**3*crust_density)/(M*(2*l+1))*sum1
                    sum0 = 0
                    sum1 = 0
                print("Degree computed: ", l)

            klm = pysh.SHGravCoeffs.from_array(klmi, gm=clm.gm, r0=clm.r0)

            file = open(savename, 'wb')
            pickle.dump(klm, file)
            file.close()

        if storage == True:
            file = open(savename, 'rb')
            klm = pickle.load(file)  # Loads the coefficients klm related to the higher order expansion
            file.close()

        ell = pysh.SHGrid.from_ellipsoid(lmax=klm.lmax,
                                         a=a,
                                         b=a - a * f)

        elm = pysh.SHGravCoeffs.from_shape(shape=ell,
                                           rho=crust_density,
                                           gm=clm.gm)

        elm = elm.change_ref(r0=clm.r0)

        ba = clm - klm + elm  # Gravity coefficients of bouguer anomaly

        # Computing Bouguer anomaly map

        ba_grid = ba.expand(a=a, f=f)  # Expansion into a grid

        if show == True:
            fig5 = (ba_grid.total*1e5).plotgmt(projection='robinson',
                                                     central_longitude=central_longitude,
                                                     cmap='roma',
                                                     cmap_limits=limits,
                                                     tick_interval=None,
                                                     cmap_reverse=True,
                                                     colorbar='bottom',
                                                     cb_label='Gravity acceleration',
                                                     cb_ylabel='mgal',
                                                     title=title,
                                                     grid=True,
                                                     shading=False)

            fig5.show()

        return ba_grid, klm
    if type == 'slab':
        topo = hlm.expand()
        G = 6.6743e-11
        bc_grid = 2*np.pi*crust_density*G*topo
        bc_array = bc_grid.expand()
        bc = pysh.SHGravCoeffs.from_array(bc_array.coeffs,gm=clm.gm,r0=clm.r0,lmax=clm.lmax)
        ell = pysh.SHGrid.from_ellipsoid(lmax=bc.lmax,
                                         a=a,
                                         b=a-a*f)

        elm = pysh.SHGravCoeffs.from_shape(shape=ell,
                                           rho=crust_density,
                                           gm=clm.gm)

        elm = elm.change_ref(r0=clm.r0)

        ba = clm - bc + elm  # Gravity coefficients of bouguer anomaly

        # Computing Bouguer anomaly map

        ba_grid = ba.expand(a=a, f=f)  # Expansion into a grid

        if show == True:
            fig5 = (ba_grid.total*1e5).plotgmt(projection='robinson',
                                                  central_longitude=central_longitude,
                                                  cmap='roma',
                                                  cmap_limits=limits,
                                                  tick_interval=None,
                                                  cmap_reverse=True,
                                                  colorbar='bottom',
                                                  cb_label='Gravity acceleration',
                                                  cb_ylabel='mgal',
                                                  title=title,
                                                  grid=True,
                                                  shading=False)

            fig5.show()
        return ba_grid, ba


def profilePlot(hlm,map,latitude,title):
    hlm.coeffs[0, 0, 0] = 0
    topo = hlm.expand() / 1000

    lat_step = 180 / map.data.shape[0]
    lon_step = 360 / map.data.shape[1]
    lat = latitude
    ground = topo.data.min()
    # numeric_lat = int(-crust_map.data.shape[0]/2+(180-lat)*lat_step)
    if lat < 0:
        lat = 90 + lat
        numeric_lat = int(180 - lat * lat_step)
    else:
        numeric_lat = int(180 - lat * lat_step)

    numeric_lat = int(map.data.shape[0] / 2)

    crust_profile = map.data[numeric_lat]
    xaxis = np.array(range(0, map.data.shape[1])) * lon_step
    trasl = crust_profile[0] - topo.data[0]
    fig, ax = plt.subplots()
    ax.plot(xaxis, 0 * np.ones((map.data.shape[1], 1)), 'r--')
    ax.fill_between(xaxis, -(crust_profile - abs(topo.data[numeric_lat])), topo.data[numeric_lat], color='grey')
    ax.fill_between(xaxis, topo.data[numeric_lat], 0, color='C1')
    ax.grid = True
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Crustal thickness [km]')
    plt.title(title)
    plt.show()

    return crust_profile, xaxis


def griddify(map,region,limits=None,):
    lat_step = 180 / map.data.shape[0]
    lon_step = 360 / map.data.shape[1]

    lat = np.array(range(0, map.data.shape[0]))
    lon = np.array(range(0, map.data.shape[1]))
    x = np.empty((0, 0))
    y = np.empty((0, 0))
    z = np.empty((0, 0))
    data = np.zeros((lat.shape[0] * lon.shape[0], 3))

    for i in range(0, lat.shape[0]):
        x = np.append(x, np.ones((1, lon.shape[0])) * (90 - lat[i] * lat_step))

    for j in range(0, lat.shape[0]):
        y = np.append(y, lon * lon_step - 180)

    for i in range(0, lat.shape[0]):
        z = np.append(z, map.data[i])

    data[:, 0] = y
    data[:, 1] = x
    data[:, 2] = z

    xspace = data[0, 1] - data[map.data.shape[1], 1]

    if region == "d":
        for k in range(0, data.shape[0]):
            if data[k][2] > limits[1]:
                data[k][2] = limits[1]
            if data[k][2] < limits[0]:
                data[k][2] = limits[0]

    grid = pygmt.xyz2grd(data=data,
                         spacing=xspace,
                         region=region)

    return grid


def plotSigma(dataCrust,samples,limits,show,title):
    file = open(dataCrust, 'rb')
    crust_total = pickle.load(file)
    file.close()

    s = np.zeros((crust_total.shape[1], crust_total.shape[2]))
    for N in range(0, samples):
        s = s + crust_total[N]

    mi = s / samples

    S = np.zeros((crust_total.shape[1], crust_total.shape[2]))
    for N in range(0, samples):
        S = S + (crust_total[N] - mi) ** 2

    sigma = np.sqrt(S / samples)
    sigma = pysh.SHGrid.from_array(sigma)

    if show == True:
        fig = sigma.plotgmt(projection='robinson',
                            central_longitude=60.,
                            cmap='plasma',
                            tick_interval=None,
                            cmap_limits=limits,
                            cmap_reverse=True,
                            colorbar='bottom',
                            cb_label='@[\sigma_{T_}@[',
                            cb_ylabel='km',
                            grid=True,
                            shading=False,
                            title=title)

        fig.show()

    return sigma, mi


def plotMoho(pot,topo,R,lmax,rho_c,rho_m,thickave,filter_type,half,nmax,delta_max,lmax_calc,correction,quiet,central_longitude,limits,show,storage,savename):
    lmax_calc = pot.lmax

    if storage==False:
        t_moho = ctp.pyMoho(pot=pot,
                        topo=topo,
                        lmax=pot.lmax,
                        rho_c=rho_c,
                        rho_m=rho_m,
                        thickave=thickave,
                        filter_type=filter_type, #0
                        half=half, #None
                        nmax=nmax,
                        delta_max=delta_max,
                        lmax_calc=lmax_calc,
                        correction=correction, #None
                        quiet=quiet)

        np.save(savename, t_moho, allow_pickle=True)

    else:
        file = open(savename, 'rb')
        t_moho = pickle.load(file)
        file.close()

    topo_calc = topo.expand(lmax=lmax_calc) / 1000 - R / 1000

    crust_map = R / 1000 - t_moho.expand() / 1000 + topo_calc  # Computing actual thickness (From center to top minus the Moho radius)

    fig1 = crust_map.plotgmt(projection='robinson',
                             central_longitude=180.,
                             cmap='roma',
                             cmap_limits=[20, 70],
                             tick_interval=None,
                             cmap_reverse=True,
                             colorbar='bottom',
                             cb_label='Crust thickness',
                             cb_ylabel='km',
                             grid=True,
                             shading=False,
                             title='Venus crustal thickness')


def gravityPerturber(Pcov,clm,lmax,iter,storage,savename):
    R = np.linalg.cholesky(Pcov)  # Cholesky decomposition
    c_tilde = np.zeros([iter, 2, lmax, lmax])

    if storage==False:

        n_coeffs = Pcov.shape[0]

        # Building of the coefficient vector. All values are stored from a 3D matrix into a vector in this shape:

        c_vec = np.zeros((1, n_coeffs), dtype=float)  # Initialization
        print("Building coefficient vector...")
        for l1 in range(2, lmax - 1):
            for m1 in range(0, lmax - 1):
                c_vec[0, l1 + m1] = clm.coeffs[0, l1, m1]
                c_vec[0, l1 + m1 + 1] = clm.coeffs[1, l1, m1]
        print("Vector built!")

        for i in range(0, iter - 1):
            if i == 0:
                print("Building gravity perturbations...")
            start = time.perf_counter()
            u = np.random.normal(size=c_vec.shape)
            P = np.transpose(R.dot(np.transpose(u)))
            c_tilde_vec = c_vec + P
            for l in range(0, lmax - 1):
                for m in range(0, lmax - 1):
                    c_tilde[i][0][l][m] = c_tilde_vec[0][m + l]
                    c_tilde[i][1][l][m] = c_tilde_vec[0][m + l + 1]
            end = time.perf_counter()
            delta = end - start
            if i == 0:
                print("Estimated ending time: ",
                      datetime.datetime.today() + datetime.timedelta(seconds=delta * iter))

        print("Perturbations completed!")
        file = open(savename, 'wb')
        pickle.dump(c_tilde, file)
        file.close()

    else:
        print("Loading gravity perturbations...")
        file = open(savename, 'rb')
        c_tilde = pickle.load(file)
        file.close()
        print("Perturbations completed!")

    return c_tilde


def densityPerturber(Pcov,rho_lm,lmax,iter,storage,savename):
    R = np.linalg.cholesky(Pcov)  # Cholesky decomposition
    r_tilde = np.zeros([iter, 2, lmax, lmax])

    if storage==False:
        n_coeffs = Pcov.shape[0]

        # Building of the coefficient vector. All values are stored from a 3D matrix into a vector in this shape:
        # [C00,0,0...,C10,C11,0,...,...,Cll,S00,0,...,...,Sll]

        r_vec = np.zeros((1, n_coeffs), dtype=float)  # Initialization
        print("Building coefficient vector...")
        for l1 in range(2, lmax - 1):
            for m1 in range(0, lmax - 1):
                r_vec[0, l1 + m1] = rho_lm.coeffs[0, l1, m1]
                r_vec[0, l1 + m1 + 1] = rho_lm.coeffs[1, l1, m1]
        print("Vector built!")

        for i in range(0, iter - 1):
            if i == 0:
                print("Building gravity perturbations...")
            start = time.perf_counter()
            u = np.random.normal(size=r_vec.shape)
            P = np.transpose(R.dot(np.transpose(u)))
            r_tilde_vec = r_vec + P
            for l in range(0, lmax - 1):
                for m in range(0, lmax - 1):
                    r_tilde[i][0][l][m] = r_tilde_vec[0][m + l]
                    r_tilde[i][1][l][m] = r_tilde_vec[0][m + l + 1]
            end = time.perf_counter()
            delta = end - start
            if i == 0:
                print("Estimated ending time: ", datetime.datetime.today() + datetime.timedelta(seconds=delta * iter))

        print("Perturbations completed!")
        file = open(savename, 'wb')
        pickle.dump(r_tilde, file)
        file.close()
    else:
        print("Loading density perturbations...")
        file = open(savename, 'rb')
        c_tilde = pickle.load(file)
        file.close()
        print("Perturbations completed!")

    return r_tilde


def txt2coeffs(txt,lmax,isGrav = False,gm=None,r0=None):
    data = np.loadtxt(txt)  # kg/m^3
    raw = np.zeros(shape=(2, lmax+1, lmax+1))
    sum_l = 0

    for l in range(0, lmax+1):
        for m in range(0, l + 1):
            raw[0][l][m] = data[sum_l + m][2]
            raw[1][l][m] = data[sum_l + m][3]
        sum_l += (l + 1)

    if isGrav == False:
        c_lm = pysh.SHCoeffs.from_array(raw)
    else:
        c_lm = pysh.SHGravCoeffs.from_array(raw,gm,r0)

    return c_lm


def pyBatoHilmRhoHV(moho, density, rho_m, ba, r0, half, lmax, M, nmax):

    t_lm = moho.expand()

    d = t_lm.coeffs[0][0][0]

    density = rho_m-density
    rholm = density.expand()

    rho_h_grid_n = (moho-d)*density

    rho_h_lm_n = rho_h_grid_n.expand()

    rho_h_lm = pysh.SHCoeffs.from_zeros(lmax=lmax)
    rhomean = 2971.495942050451

    # First term

    lambda_l = 1 / ((M * (2 * half + 1) / (4 * np.pi * (rho_m - rhomean) * d ** 2)) * (r0 / d) ** (half)) ** 2
    for l in range(1, lmax + 1):
        w_l = 1 / (1 + lambda_l * (
                (M * (2 * l + 1) / (4 * np.pi * (rho_m - rhomean) * d ** 2)) * (r0 / d) ** l) ** 2)
        for m in range(1, l + 1):
            rho_h_lm.coeffs[0][l][m] = w_l * ba[0][l][m] * M * (2 * l + 1) * ((r0 / d) ** l) / (4 * np.pi * d ** 2)
            rho_h_lm.coeffs[1][l][m] = w_l * ba[1][l][m] * M * (2 * l + 1) * ((r0 / d) ** l) / (4 * np.pi * d ** 2)

    # Higher terms

    for n in range(1, nmax):
        rho_h_grid_n = density * ((moho - d) / d) ** n
        rho_h_lm_n = rho_h_grid_n.expand()

        for l in range(1, lmax):
            prod = 1
            for j in range(1, n):
                prod = prod * (l + 4 - j)
            prod = d * prod / ((l + 3) * scipy.special.factorial(n))
            for m in range(0, l + 1):
                rho_h_lm.coeffs[0][l][m] = rho_h_lm.coeffs[0][l][m] - w_l * rho_h_lm_n.coeffs[0][l][m] * prod
                rho_h_lm.coeffs[1][l][m] = rho_h_lm.coeffs[1][l][m] - w_l * rho_h_lm_n.coeffs[1][l][m] * prod

    rho_h_grid = rho_h_lm.expand(grid='DH2', lmax=lmax, extend=False)
    h_grid = rho_h_grid / density
    t_lm = h_grid.expand()
    t_lm.coeffs[0][0][0] = d

    return t_lm.coeffs


def pyMohoRhoV(flag, pot, topo, rholm, rho_0, delta_rho, porosity, lmax, rho_m, thickave,
              filter_type=0, half=None, nmax=8, delta_max=5., lmax_calc=None,
              correction=None, quiet=False):

    quant = 0

    R = topo.coeffs[0][0][0] # Mean radius

    if (filter_type == 1 or filter_type == 2) and half is None:
        raise ValueError("half must be set when filter_type is either 1 or 2.")

    if lmax_calc is None:
        lmax_calc = lmax

    density_original = rho_0+delta_rho*thickave # Density corrected with depth
    density = density_original.expand()  # Density coefficients
    density = rholm

    d = R - thickave # Moho boundary
    rho_crust_ave = density.coeffs[0, 0, 0] * (1. - porosity)

    mass = pot.mass

    topo_grid = topo.expand(grid='DH2', lmax=lmax, extend=False)
    density_grid = density.expand(grid='DH2', lmax=lmax, extend=False) # back to grid

    if quiet is False:
        print("Maximum radius (km) = {:f}".format(topo_grid.data.max() / 1.e3))
        print("Minimum radius (km) = {:f}".format(topo_grid.data.min() / 1.e3))
        print("Maximum density (kg/m3) = {:f}".format(
            density_grid.data.max() / 1.e3))
        print("Minimum desntiy (kg/m3) = {:f}".format(
            density_grid.data.min() / 1.e3))

    bc, r0 = pysh.gravmag.CilmPlusRhoHDH(
        topo_grid.data, nmax, mass, density_grid.data * (1. - porosity),
        lmax=lmax_calc) # bouguer and mean radius of bouguer
    if correction is not None:
        bc += correction.change_ref(r0=r0).to_array(lmax=lmax_calc)

    pot2 = pot.change_ref(r0=r0)
    ba = pot2.to_array(lmax=lmax_calc, errors=False) - bc

    # next subtract lateral variations in the crust without reflief
    for l in range(1, lmax_calc + 1):
        ba[:, l, :l + 1] = ba[:, l, :l + 1] \
                           - 4. * np.pi * density.coeffs[:, l, :l + 1] \
                           * (1. - porosity) \
                           * (r0**3 - (d**3)*(d/r0)**l) \
                           / (2 * l + 1) / (l + 3) / mass

    moho = pysh.SHCoeffs.from_zeros(lmax=lmax_calc)
    moho.coeffs[0, 0, 0] = d

    for l in range(1, lmax_calc + 1):
        if filter_type == 0:
            moho.coeffs[:, l, :l + 1] = ba[:, l, :l + 1] * mass * \
                (2 * l + 1) * ((r0 / d)**l) / \
                (4. * np.pi * (rho_m - rho_crust_ave) * d**2)
        elif filter_type == 1:
            moho.coeffs[:, l, :l + 1] = pysh.gravmag.DownContFilterMA(
                l, half, r0, d) * ba[:, l, :l + 1] * mass * \
                (2 * l + 1) * ((r0 / d)**l) / \
                (4. * np.pi * (rho_m - rho_crust_ave) * d**2)
        else:
            moho.coeffs[:, l, :l + 1] = pysh.gravmag.DownContFilterMC(
                l, half, r0, d) * ba[:, l, :l + 1] * mass * \
                (2 * l + 1) * ((r0 / d)**l) / \
                (4.0 * np.pi * (rho_m - rho_crust_ave) * d**2)

    moho_grid3 = moho.expand(grid='DH2', lmax=lmax, lmax_calc=lmax_calc,
                             extend=False)

    density_grid = rho_0+delta_rho*(R-moho_grid3)

    temp_grid = topo_grid - moho_grid3

    if quiet is False:
        print('Maximum Crustal thickness (km) = {:f}'.format(
            temp_grid.data.max() / 1.e3))
        print('Minimum Crustal thickness (km) = {:f}'.format(
            temp_grid.data.min() / 1.e3))


    moho.coeffs = pyBatoHilmRhoHV(moho_grid3, half = half, lmax=lmax, density= density_grid, rho_m = rho_m, ba=ba, r0=r0, M=mass, nmax=nmax)
    moho_grid2 = moho.expand(grid='DH2', lmax=lmax, lmax_calc=lmax_calc,
                             extend=False)

    temp_grid = topo_grid - moho_grid2

    if quiet is False:
        print('Delta (km) = {:e}'.format(abs(moho_grid3.data -
                                             moho_grid2.data).max() / 1.e3))
        print('Maximum Crustal thickness (km) = {:f}'
              .format(temp_grid.data.max() / 1.e3))
        print('Minimum Crustal thickness (km) = {:f}'
              .format(temp_grid.data.min() / 1.e3))

    iter = 0
    delta = 1.0e9

    flag = 0

    while delta > delta_max:
        if flag == 0:

            iter += 1

            if quiet is False:
                print('Iteration {:d}'.format(iter))

            moho_grid = (moho_grid2 + moho_grid3) / 2.
            temp_grid = topo_grid - moho_grid

            if quiet is False:
                print("Delta (km) = {:e}".format(
                    abs(moho_grid.data - moho_grid2.data).max() / 1.e3))
                print('Maximum Crustal thickness (km) = {:e}'.format(
                    temp_grid.data.max() / 1.e3))
                print('Minimum Crustal thickness (km) = {:e}'.format(
                    temp_grid.data.min() / 1.e3))

            moho_grid3 = moho_grid2
            moho_grid2 = moho_grid

            iter += 1

            if quiet is False:
                print('Iteration {:d}'.format(iter))

            density_grid = rho_0 + delta_rho * (R - moho_grid3)

            moho.coeffs = pyBatoHilmRhoHV(moho_grid3, half=half, lmax=lmax, density=density_grid, rho_m=rho_m, ba=ba,
                                          r0=r0, M=mass, nmax=nmax)
            moho_grid = moho.expand(grid='DH2', lmax=lmax, lmax_calc=lmax_calc,
                                    extend=False)

            delta = abs(moho_grid.data - moho_grid2.data).max()
            temp_grid = topo_grid - moho_grid

            if quiet is False:
                print('Delta (km) = {:e}'.format(delta / 1.e3))
                print('Maximum Crustal thickness (km) = {:f}'.format(
                    temp_grid.data.max() / 1.e3))
                print('Minimum Crustal thickness (km) = {:f}'.format(
                    temp_grid.data.min() / 1.e3))

            moho_grid3 = moho_grid2
            moho_grid2 = moho_grid

            if abs(temp_grid.data).max() > 500e3:
                print('ohhhps!')
                #for v in range(0, temp_grid.data.shape[0]):
                #    for b in range(0,temp_grid.data.shape[1]):
                #        if temp_grid.data[v][b]>500e3:
                #            temp_grid.data[v][b] = 500e3
                flag = 1
                return moho, flag
            else:
                flag = 0

    return moho, flag
