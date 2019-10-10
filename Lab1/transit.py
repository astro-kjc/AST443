import numpy as np
import matplotlib.pyplot as plt
import os

class Star:

    starcoords = np.loadtxt("lightcurves/ref_stars.csv", delimiter=",")
    times = np.loadtxt("lightcurves/times.csv", delimiter=",")

    def __init__(self, index):

        directory = f"lightcurves/star_{index}"
        self.ra, self.dec = self.starcoords[index]
        self.flux = np.loadtxt(os.path.join(directory, f"flux_{index}.csv"), delimiter=",")
        self.flux_err = np.loadtxt(os.path.join(directory, f"flux_err_{index}.csv"), delimiter=",")
        self.indices = np.loadtxt(os.path.join(directory, f"indices_{index}.csv"), dtype=np.int32, delimiter=",")

    def __str__(self):

        return f"Star(RA={self.ra:.3f}, Dec={self.dec:.3f})"

    def trim(self, sigma_tol=3):

        mu_flux = self.flux.mean()
        std_flux = self.flux.std()
        num_sd = np.abs(self.flux - mu_flux) / std_flux

        mask = num_sd <= sigma_tol
        self.flux = self.flux[mask]
        self.flux_err = self.flux_err[mask]
        self.indices = self.indices[mask]

    def plot(self, binsize=None, figsize=(12, 9)):

        if binsize is None:
            times = self.times[self.indices]
            flux = self.flux
            flux_err = self.flux_err
        else:
            times = np.arange(Star.times[0] + binsize/2, Star.times[-1] - binsize/2, step=binsize)
            flux, flux_err = self.bin_fluxes(binsize)
            flux = flux[:len(times)]
            flux_err = flux_err[:len(times)]
            
        label = rf"$\alpha$ = {self.ra:.2f}$^\circ$, $\delta$ = {self.dec:.2f}$^\circ$"
        plt.errorbar(times / 3600, flux, flux_err, fmt='.',
                    label=label, markersize=8, capsize=2)
        plt.xlabel("Time Since Midnight UTC [h]")
        plt.ylabel(r"$\frac{f}{\langle f \rangle}$")
        plt.legend()
        plt.gcf().set_size_inches(figsize)
        plt.show()

    def bin_fluxes(self, binsize):

        nbins = (Star.times[-1] - Star.times[0]) / binsize
        scalefac = nbins / len(Star.times)
        bindices = np.floor(self.indices * scalefac)
        bindices = bindices.astype(np.int32)

        binned_flux = np.zeros(bindices.max() + 1)
        exp_flux2 = np.zeros(len(binned_flux))
        counts = np.zeros(len(binned_flux), dtype=np.int32)
        
        for i in range(len(self.flux)):
            
            binned_flux[bindices[i]] += self.flux[i]
            exp_flux2[bindices[i]] += self.flux[i]**2
            counts[bindices[i]] += 1

        binned_flux /= counts
        exp_flux2 /= counts
        sigma = np.sqrt((exp_flux2 - binned_flux**2) / counts)

        return binned_flux, sigma

def keep_frames_with_all(stars):

    counts = np.zeros(len(Star.times), dtype=np.int32)

    for star in stars:
        counts[star.indices] += 1

    mask = counts == len(stars)
    indices = np.arange(0, len(Star.times), dtype=np.int32)
    indices = indices[mask]
    indices = set(indices)
    
    @np.vectorize
    def valid_indices(i):
        return i in indices
    
    for star in stars:

        mask = valid_indices(star.indices)
        star.flux = star.flux[mask]
        star.flux_err = star.flux_err[mask]
        star.indices = star.indices[mask]

def err_weighted_mean(stars):

    mu = np.zeros(len(Star.times))
    sigma = np.zeros(len(Star.times))

    for star in stars:

        mu[star.indices] += star.flux / star.flux_err**2
        sigma[star.indices] += 1 / star.flux_err**2

    mu[mu == 0.0] = 1.0
    sigma[sigma == 0.0] = 1.0

    mu /= sigma
    sigma = np.sqrt(1 / sigma)

    return mu, sigma

if __name__ == "__main__":
    
    main_star = Star(0)
    ref_stars = list(map(Star, range(1, 11)))

    main_star.trim()
    for star in ref_stars:
        star.trim()

    keep_frames_with_all([main_star] + ref_stars)

    mu, sigma = err_weighted_mean(ref_stars)
    main_star.flux = main_star.flux / mu[main_star.indices]
    # Need to account for error in mu
    main_star.flux_err = main_star.flux_err / mu[main_star.indices]

    baseline_mask = Star.times[main_star.indices] < 9000
    baseline_mask |= Star.times[main_star.indices] > 16000

    # Do a linear fit - there is a slight increase in flux
    baseline_times = Star.times[main_star.indices][baseline_mask]
    baseline_flux = main_star.flux[baseline_mask]
    m, b = np.polyfit(baseline_times, baseline_flux, 1)
    baseline = m*Star.times[main_star.indices] + b

    main_star.flux /= baseline
    main_star.flux_err /= baseline
    
    binsize = 180
    main_star.plot(binsize)
    time_bounds = Star.times[0] + binsize/2, Star.times[-1] - binsize/2,
    times = np.arange(*time_bounds, step=binsize)
    binned_flux, binned_flux_err = main_star.bin_fluxes(binsize)
    binned_flux = binned_flux[:len(times)]
    binned_flux_err = binned_flux_err[:len(times)]

    def part_of_dip(i):
        meets_thresh = binned_flux[i] <= 0.995
        left_mt = right_mt = False
        if i > 0: left_mt = binned_flux[i-1] <= 0.995
        if i < len(times)-1: right_mt = binned_flux[i+1] <= 0.995
        return meets_thresh & (left_mt | right_mt)
    transit_mask = list(map(part_of_dip, range(len(binned_flux))))
    transit_times = times[transit_mask]
    transit_flux = binned_flux[transit_mask]

    # Compute transit midpoint
    midpoint = np.median(transit_times)
    minutes = midpoint // 60 % 60
    hours = (midpoint // 3600 - 4) % 24
    print(f"Midpoint: {hours:.0f}:{minutes:.0f} +- 00:03")

    # Compute transit depth
    transit_depth = 1 - transit_flux.mean()
    print(f"Transit Depth: {transit_depth*100:.1f}%")

    # Planet radius
    rad_rat = np.sqrt(transit_depth)
    print(f"Radius Ratio: {rad_rat:.3f}")
