import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import timeit
from astropy.cosmology import WMAP9 as cosmo

def get_xyz(filename):
    f = fits.open(filename)
    data = f[1].data
    header = f[1].header
    ra,dec,redshift = data["RA"],data["DEC"],data["Z"]

    d = cosmo.comoving_distance(redshift)

    #plt.plot(ra,dec,".",markersize=1)
    #plt.show()

    d = d.value
    idx = (d>0) # removes only 10 galaxies
    ra,dec,d = ra[idx],dec[idx],d[idx]

    #plt.plot(ra,dec,".",markersize=1)
    #plt.show()

    conv = np.pi / 180
    x = d*np.cos(conv*ra)*np.cos(conv*dec)
    y = d*np.sin(conv*ra)*np.cos(conv*dec)
    z = d*np.sin(conv*dec)
    print(len(x))

    return (x,y,z)

x,y,z = get_xyz("ELG_N_clustering.dat.fits")
# once the sample is selected, will have 3 arrays: x,y,z

# choosing bins:
# bins = 10 ** np.linspace(np.log10(1.0/10), np.log10(20), 16)
bins = np.linspace(1.0/10, 200, 41)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

t_start = timeit.default_timer() # starting timer

# calculating separation between all objects:
n = len(x) # number of objects

# DD:
sep = np.zeros(int(n*(n-1)/2)) - 1
previous = 0
for i in range(n):
    dist = np.sqrt( (x[i]-x[i+1:n])**2 + (y[i]-y[i+1:n])**2 + (z[i]-z[i+1:n])**2)
    dist = dist[dist<200]
    sep[previous:previous+len(dist)] = dist.copy()
    previous += len(dist)
print("Done with DD seps")

sep= sep[sep != -1]
dd = np.zeros(len(bins)-1)
for i in range(len(bins)-1):
    left = bins[i]
    right = bins[i+1]
    dd[i] = np.sum((sep>left) & (sep<right))

DD = dd / (n*(n-1)/2)

# averaging dr and rr over multiple random datasets:

DR,RR = np.zeros_like(DD), np.zeros_like(DD)
num_rsamples = 5
for num in range(5,10):
    randx,randy,randz = get_xyz("ELG_N_" + str(num) + "_clustering.ran.fits")
    randn = len(randx)

    # DR:
    sep = np.zeros(int(n*randn)) - 1
    previous = 0
    for i in range(n):
        dist = np.sqrt( (x[i]-randx)**2 + (y[i]-randy)**2 + (z[i]-randz)**2)
        dist = dist[dist<200]
        sep[previous:previous+len(dist)] = dist.copy()
        previous += len(dist)
    print("Done with DR seps")

    sep= sep[sep != -1]
    dr = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        left = bins[i]
        right = bins[i+1]
        dr[i] = np.sum((sep>left) & (sep<right))
        
    DR += (dr / randn / n)


    # RR:
    sep = np.zeros(int(randn*(randn-1)/2)) - 1
    previous = 0
    for i in range(randn):
        dist = np.sqrt( (randx[i]-randx[i+1:randn])**2 + (randy[i]-randy[i+1:randn])**2 + (randz[i]-randz[i+1:randn])**2)
        dist = dist[dist<200]
        sep[previous:previous+len(dist)] = dist.copy()
        previous += len(dist)
    print("Done with RR seps")

    sep= sep[sep != -1]
    rr = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        left = bins[i]
        right = bins[i+1]
        rr[i] = np.sum((sep>left) & (sep<right))
        
    RR += (rr / (randn*(randn-1)/2))

DR = DR / num_rsamples
RR = RR / num_rsamples

t_stop = timeit.default_timer() # stopping the timer
print("Seconds to calculate separations: ", t_stop-t_start)

# calculate Landy-Szalay estimator for 2pcf:
xi = (DD - 2*DR + RR) / RR
print(xi)


# plot results:
plt.plot(bin_centers, xi, ".")
# plt.plot(bin_centers, bin_centers**2*xi, ".")
# plt.xscale("log") # log scale
# plt.yscale("log")
# plt.ylim(-0.05,0.05)
plt.xlabel(r"Separation $r$ [Mpc]", fontsize=15)
plt.ylabel(r"$\xi(r)$", fontsize=15)
plt.title("DESI ELG")
plt.show()


with open("corr_ELG_to200Mpc.txt", "w") as f:
    f.write("#number of galaxies: "+str(len(x))+"\n")
    for a in bin_centers:
        f.write(str(a)+" ")
    f.write("\n")
    for a in xi:
        f.write(str(a)+" ")
f.close