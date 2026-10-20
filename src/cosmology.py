from astropy import constants as const
from astropy import units as u
import numpy as np

class ReadCosmology:
    """
    Calculate basic cosmological quantities for a ΛCDM universe.
    The default parameters are based on Planck Collaboration XVI results.
    
    Parameters
    ----------

    z: float, optional
        Redshift at which to evaluate the cosmological quantities.
        Default value is 0.0

    h: float, optional
        Dimensionless hubble parameter, where H0 = 100 h km / s / Mpc
        Default is 0.673

    Omega_m: float, optional
        Present-day total matter density parameter.
        Default is 0.315

    Omega_b_h2: float, optional
        Present-day baryonic density parameter, Omega_b * h**2.
        Default is 0.02205

    X: float, optional
        Hydrogen mass fraction.
        Default is 0.76

    Returns
    -------
        dictionary:
            Constants:
                h: Dimensionless Hubble parameter
                H0: Present-day Hubble constant
                Omega_m: Present-day matter density parameter
                Omega_b: Present-day baryon density parameter
                X: Hydrogen mass fraction

            Quantities:
                H: Hubble parameter at redshift z
                rhocrit: Critical density at redshift z
    """  
    def __init__(self, z: float = 3.0, h: float = 0.673, Omega_m: float = 0.315, Omega_b_h2: float = 0.02205, X: float = 0.76) -> None:
        
        if not np.isfinite(z) or z < 0:
            raise ValueError("Redshift z must be finite and non-negative.")
    
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive.")
    
        if not np.isfinite(Omega_m) or not 0 < Omega_m <= 1:
            raise ValueError("Omega_m must be in the interval (0, 1].")
    
        if not np.isfinite(Omega_b_h2) or Omega_b_h2 <= 0:
            raise ValueError("omega_b_h2 must be finite and positive.")
    
        if Omega_b_h2 * h**2 > Omega_m:
            raise ValueError("Omega_b cannot exceed Omega_m.")
    
        if not np.isfinite(X) or not 0 <= X <= 1:
            raise ValueError("X must be between 0 and 1.")

        self.z          = z
        self.h          = h
        self.H0         = 100 * h * u.km / u.s / u.Mpc
        self.Omegam     = Omega_m
        self.Omegab     = Omega_b_h2 / h**2
        self.X          = X
        
    def compute_hubble(self) -> u.Quantity:
        """Return the Hubble constant at redshift z.

        Returns
        -------
        astropy.units.Quantity
            H(z), in 1/s
        """
        
        H = self.H0 * np.sqrt(
            self.Omegam*(1 + self.z)**3
             + (1-self.Omegam)
        )
        
        return H.to(1 / u.s)

    def compute_rhocrit(self) -> u.Quantity:
        """Return the critical densiy at redshift z.

        Returns
        -------
        astropy.units.Quantity
            Critical density, in g/cm^3
        
        """
        H = self.compute_hubble()
        
        rhocrit = (3 * H**2) / (8 * np.pi * const.G)
        
        return rhocrit.to(u.g / u.cm**3)

    def read_cosmo(self) -> dict:
        """Return cosmological constants and quantities in a dictionary.

        Returns
        -------
        dict
            A dictionary with 'Constants' and 'Quantities' entries.
    
        """
    
        cosmo_dict = {
            "Constants": {
                "h": self.h,
                "H0": self.H0,
                "Omegam": self.Omegam,
                "Omegab": self.Omegab,
                "X": self.X
            },
            "Quantities": {
                "H": self.compute_hubble(),
                "rhocrit": self.compute_rhocrit()
            }
        }

        return cosmo_dict