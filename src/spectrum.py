from scipy.integrate import quad
from scipy.stats import norm
import numpy as np
import multiprocessing as mp
import astropy.units as u
from astropy import constants as const
from tqdm import tqdm

from cosmology import ReadCosmology
from eagle_query import initiate_query

def _init_worker(processor):
    global _worker_processor
    _worker_processor = processor

def _compute_spectrum_worker(args):
    doppler, sightline_pos = args
    return _worker_processor._calc_single_spectrum(doppler = doppler, sightline_pos = sightline_pos)

class Spectrum:
    def __init__(self, username: str,
                 password: str, 
                 simName: str = "L0100N1504", 
                 snapNum: int = 12,
                 fgas: float = 0.6, 
                 alpha: float = 2,
                 sightline_axis: int = 2
                ) -> None:

        """Initialise the spectrum using an EAGLE snapshot"""

        if not np.isfinite(fgas) or not 0 <= fgas <= 1:
            raise ValueError("fgas must be between 0 and 1")

        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")

        if sightline_axis not in (0, 1, 2):
            raise ValueError("sightline_axis must be 0, 1 or 2")

        self.subhalos     = initiate_query(username, password, simName, snapNum)
        self.num_subhalos = np.shape(self.subhalos)[0]

        self.z        = self.subhalos['redshift'][0]
        self.fgas     = fgas
        self.scaleFac = 1 / (1 + self.z)
        self.alpha    = alpha

        self.cosmology  = ReadCosmology(z = self.z)
        self.cosmo_dict = self.cosmology.read_cosmo()

        self.boxSizeCo   = 100 * u.Mpc
        self.boxSizePr   = self.boxSizeCo * self.scaleFac

        self.sightline_axis = sightline_axis

        self.integral, self.integral_error = quad(       # Integral to calculate the column density.
            lambda q: 1 / (1 + q**2)**self.alpha, 
            0, 
            float("inf")
        )

        if not np.isfinite(self.integral):
            raise ValueError("The integral is not finite.")

        self.nHh = self._compute_nHh()      # Hydrogen number density at the virial radius in cm^-3.
        
    def _compute_nHh(self) -> u.Quantity:
        """
        Calculate the hydrogen number density at the halo virial radius.
    
        The density is assumed to be the same for every halo and is derived
        from the mean hydrogen density within the virial radius, assuming
        a halo overdensity of 200 relative to the critical density.
    
        The hydrogen density at the virial radius is determined using the
        power-law density profile specified by self.alpha.
    
        Returns
        -------
        nHh : astropy.units.Quantity
            Hydrogen number density at the virial radius, in cm^-3.
        """
        rhocritz = self.cosmo_dict['Quantities']['rhocrit']
        Omegab   = self.cosmo_dict['Constants']['Omegab']
        Omegam   = self.cosmo_dict['Constants']['Omegam']
        X        = self.cosmo_dict['Constants']['X']
        fgas     = self.fgas
    
        omegab = Omegab / Omegam
    
        rhomeanH_h = 200 * rhocritz * X * fgas * omegab
        rhoH_h = ((3 - self.alpha) / 3) * rhomeanH_h
        nHh = (rhoH_h / const.m_p).to(1 / u.cm**3)
    
        return nHh.to(1 / u.cm**3)

    def _compute_NHI(self, 
                    R_h: u.Quantity = 100 * u.kpc, 
                    b: u.Quantity = 1 * u.Mpc, 
                    Gamma0: u.Quantity = None, 
                    alphaB: u.Quantity = None
                   ) -> u.Quantity:
        """
        Calculates the column density for each selected halo using the corresponding
        impact parameter.

        Parameters
        ----------
        R_h: astropy.units.Quantity, optional
            The virial radius of the selected halo.
            Default is 100 kpc

        b: astropy.units.Quantity, optional
            The impact prameter of the selected halo.
            Default is 1 Mpc.

        Gamma0: astropy.units.Quantity, optional
            The neutral hydrogen photoionization rate. [1/s]
            If None, the default value is used.

        alphaB: astropy.units.Quantity, optional
            The Case-B recombination coefficient for hydrogen. [cm^3/s]
            If None, the default value is used.

        Returns
        -------
        NHIs: astropy.units.Quantity, [cm^-2]
            Quantity array containing the column density for each selected halo.
        
        """
        
        # Normalized parameters
        alphaB_norm = 3e-13 * u.cm**3 / u.s    # Normalized to T = 1.5x10^4 K 
        Gamma0_norm = 0.882e-12 / u.s          # Normalized to z = 3.0

        nH_h     = self.nHh
        alpha    = self.alpha

        I = self.integral

        if Gamma0 is None:
            Gamma0 = Gamma0_norm

        if alphaB is None:
            alphaB = alphaB_norm

        NHI_h = 2 * (alphaB / Gamma0) * nH_h**2 * R_h * I

        NHIs = (NHI_h * (R_h / b)**(2*alpha - 1)).to(u.cm**-2)

        return NHIs.to(u.cm**-2)
        
    def _generate_sightline_pos(self):
        """
        Method to generate a random sightline starting position within the simulation volume.
        The sightline starts at axis coordinate zero

        Returns
        -------
        astropy.units.Quantity, shape (1, 3) [pMpc]
        
        """

        # Axis sightline travels along
        sightline_axis = self.sightline_axis
        
        # Generate random sightline starting coordinate 
        init_coords = np.random.rand(3) * self.boxSizePr

        # Fix axis coordinate to zero (arbitrary)
        init_coords[sightline_axis] = 0.0 * u.Mpc

        return init_coords.to(u.Mpc)

    def _calc_periodic_distance(self, delta_r: u.Quantity, boxsize: u.Quantity) -> u.Quantity:
        """
        Calculate displacement vectors using periodic boundary conditions.

        Parameters
        ----------
        delta_r: astropy.units.Quantity, required
            Displacement vector(s) between positions with units of length.

        boxsize: astropy.units.Quantity, required
            Boxsize of the EAGLE simulation, with compatiable unit of length.

        Returns
        -------
        periodic_delta_r: astropy.units.Quantity
            Displacement vector(s) adjusted to the minimum-image convention,
            such that each component lies within [-boxsize/2, boxsize/2).
        
        """
        periodic_delta_r = delta_r - (boxsize * np.floor((delta_r / boxsize) + 0.5))
        
        return periodic_delta_r.to(u.Mpc)
        

    def _compute_impact_params(self, sightline_pos) -> u.Quantity:
        """
        Return impact parameters from one sightline position to all halo centres.
    
        Parameters
        ----------
        sightline_pos : astropy.units.Quantity, shape (3,)
            Sightline starting coordinates in proper Mpc.
    
        Returns
        -------
        astropy.units.Quantity, shape (num_subhalos,)
            Impact parameters in Mpc.
        """

        sightline_pos = u.Quantity(sightline_pos).to(u.Mpc)

        # Halo position within simulation volume [cMpc]
        halo_pos_co = np.column_stack((
            self.subhalos['x'],
            self.subhalos['y'],
            self.subhalos['z']
        )) * u.Mpc 
        
        # Halo position within simulation volume [pMpc]
        halo_pos_pr = halo_pos_co * self.scaleFac
    
        # Displacement vectors from sightline starting pos to halo pos [pMpc]
        delta_r          = halo_pos_pr - sightline_pos
        delta_r_periodic = self._calc_periodic_distance(delta_r, self.boxSizePr)

        # Select the two perpendicular axes
        possible_axes = [i for i in range(3) if i != self.sightline_axis]

        # Calculate the impact parameters
        bs = np.linalg.norm(delta_r_periodic[:, possible_axes], axis = 1)

        assert bs.shape == (len(self.subhalos['x']),), "Impact parameter array is not 1-dimensional"
    
        return bs.to(u.Mpc)
    
    def _calc_halo_vels(
        self,
        sightline_pos: u.Quantity
    ) -> u.Quantity:
        """
        Calculate the line-of-sight velocity of each halo relative to a sightline.
    
        The total velocity combines the Hubble-flow velocity, calculated from
        the periodic line-of-sight displacement, and the halo's peculiar
        velocity along the line-of-sight axis.
    
        Parameters
        ----------
        sightline_pos : astropy.units.Quantity
            Position of the sightline in the simulation box, with units of
            length.
    
        Returns
        -------
        total_los_halo_vel : astropy.units.Quantity
            Total line-of-sight velocity of each halo, including Hubble flow
            and peculiar velocity, in km/s.
        """
        # Select the line-of-sight axis
        sightline_axis = self.sightline_axis
        axis = ['x', 'y', 'z'][sightline_axis]
        vel_axis = ['vx', 'vy', 'vz'][sightline_axis]
    
        # Select line-of-sight coordinates
        los_halo_pos_pr = self.subhalos[axis] * u.Mpc * self.scaleFac
        los_sightline_pos = sightline_pos[sightline_axis]
    
        # Calculate halo displacement, accounting for periodic boundaries
        los_halo_displacement = los_halo_pos_pr - los_sightline_pos
        los_halo_displacement_periodic = self._calc_periodic_distance(
            los_halo_displacement, self.boxSizePr
        )
    
        # Calculate Hubble-flow and peculiar velocities
        hubble_vel = (
            self.cosmo_dict['Quantities']['H'] * los_halo_displacement_periodic
        )
        los_halo_vel = self.subhalos[vel_axis] * u.km / u.s
    
        total_los_halo_vel = hubble_vel + los_halo_vel
    
        return total_los_halo_vel.to(u.km / u.s)

    def _calc_single_spectrum(self, 
                             doppler: u.Quantity = 23 * u.km / u.s,
                             vel_res: u.Quantity = 1 * u.km / u.s, 
                             sightline_pos: u.Quantity = None
                            ) -> tuple:
        """
        Calculate the optical depth and transmitted flux along a single sightline.

        Parameters
        ----------
        doppler: astropy.units.Quantity, optional
            Doppler parameter to describe thermal motion of the hydrogen atoms. [km/s]
            Default is 23 km/s

        vel_res: astropy.units.Quantity, optional
            Increment of velocity grid. [km/s]
            Default is 1 km/s

        sightline_pos: astropy.units.Quantity, optional
            Starting position for the sightline.
            Default is None
            
        Returns
        -------
        NHIs: astropy.units.Quantity
            Neutral hydrogen column density for each selected halo. [cm^-2]

        vel_grid: astropy.units.Quantity
            Velocity grid for which the spectrum is evaluated on. [km/s]

        flux: numpy.ndarray
            Transmitted flux as a function of velocity.

        tau_v: numpy.ndarray
            Optical depth as a function of velocity.
            
        """
        
        doppler_value = u.Quantity(doppler).to_value(u.km / u.s)
        vel_res_value = u.Quantity(vel_res).to_value(u.km / u.s)
        
        if not np.isfinite(doppler_value) or doppler_value <= 0:
            raise ValueError("doppler must be finite and positive.")
        
        if not np.isfinite(vel_res_value) or vel_res_value <= 0:
            raise ValueError("vel_res must be finite and positive.")

        if sightline_pos is None:
            sightline_pos = self._generate_sightline_pos()

        impact_params = self._compute_impact_params(sightline_pos)

        in_range      = impact_params < 5 * u.Mpc              # Limit maximum impact parameter to remove minimally contributing halos
        R_hs_in_range = self.subhalos['r200'][in_range] * u.kpc
        bs_in_range   = impact_params[in_range]
        NHIs          = self._compute_NHI(R_h = R_hs_in_range, b = bs_in_range)

        los_halo_vels_in_range = self._calc_halo_vels(sightline_pos)[in_range]      
    
        vel_max  = (self.cosmo_dict['Quantities']['H'] * self.boxSizePr).to(u.km / u.s)       
        vel_grid = np.arange(0, vel_max.to_value(u.km / u.s), vel_res.to_value(u.km / u.s)) * u.km / u.s   

        lambda0       = 1215.67 * u.Angstrom
        f             = 0.4162
        sigma         = (np.sqrt(3 * np.pi * const.sigma_T / 8) * lambda0 * f)
        sigma_v       = doppler / np.sqrt(2)

        tau_v = np.zeros(len(vel_grid), dtype = np.float64)
        
        
        for v0, NHI in zip(los_halo_vels_in_range, NHIs):       # Loop through each halo's velocity and column density to compute optical depth at each velocity pixel.
            
            dv          = vel_grid - v0
            dv_periodic = dv - vel_max * np.floor((dv / vel_max) + 0.5)

            phi_v = norm.pdf(
                dv_periodic.to_value(u.km / u.s),
                loc = 0.0,
                scale = sigma_v.to_value(u.km / u.s),
            ) * u.s / u.km
            
            # Check gaussian profile normalization
            integral_phi = np.sum((phi_v * vel_res).to_value(u.dimensionless_unscaled))
            if not np.isclose(integral_phi, 1.0, atol=1e-2):
                print(f"Warning: Gaussian not normalized! Integral = {integral_phi:.5f}")

            tau_v += (NHI * sigma * const.c * phi_v).to(u.dimensionless_unscaled)

        flux = np.exp(-tau_v)

        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0)
        assert np.all(flux <= 1)

        return NHIs, vel_grid, flux, tau_v

    def calc_all_spectrum(self, doppler, num_spectra, ncpu):
        """
        Generate multiple absorption spectra using multiprocessing.
    
        Randomly generates sightline positions within the simulation box,
        computes a spectrum for each sightline in parallel, and collects
        the results into a dictionary.
    
        Parameters
        ----------
        doppler : astropy.units.Quantity
            Doppler parameter used to calculate the absorption profiles.
        num_spectra : int
            Number of independent sightlines and spectra to generate.
            Must be a positive integer.
        ncpu : int
            Number of worker processes to use. Must be a positive integer.
    
        Returns
        -------
        vel_grid : astropy.units.Quantity
            Velocity grid shared by all spectra, in km/s.
        result_dict : dict
            Dictionary indexed by sightline number. Each entry contains:
            - 'NHI': Neutral hydrogen column densities for the halos.
            - 'Flux': Transmitted flux spectrum.
            - 'Tau': Optical depth spectrum.
        """

        doppler_value = u.Quantity(doppler).to_value(u.km / u.s)
        
        if not np.isfinite(doppler_value) or doppler_value <= 0:
            raise ValueError("doppler must be finite and positive.")
       
        if not isinstance(num_spectra, (int, np.integer)) or isinstance(num_spectra, (bool, np.bool_)) or num_spectra < 1:
            raise ValueError("num_spectra must be a positive integer.")
        
        if not isinstance(ncpu, (int, np.integer)) or isinstance(ncpu, (bool, np.bool_)) or ncpu < 1:
            raise ValueError("ncpu must be a positive integer.")

        sightline_positions = [self._generate_sightline_pos() for _ in range(num_spectra)]

        # Check if any sightline spawn outside the box
        positions = u.Quantity(sightline_positions)
        assert np.all(positions <= self.boxSizePr) and np.all(positions >= 0 * u.Mpc), \
            "Sightline spawns outside of box"

        # List containing arguments for each sightline
        tasks = [(doppler, pos) for pos in sightline_positions]

        # Assigns workers to individual spectra
        with mp.Pool(processes = ncpu, initializer = _init_worker, initargs = (self, )) as pool:
            results = list(tqdm(pool.imap(_compute_spectrum_worker, tasks), total = num_spectra))

        # Extract the velocity grid once
        vel_grid = results[0][1]

        result_dict = {idx: {'NHI': NHIs, 'Flux': flux, 'Tau': tau_v} for idx, (NHIs, _, flux, tau_v) in enumerate(results)}

        print(f"Computed all {num_spectra} spectra!")

        return vel_grid, result_dict