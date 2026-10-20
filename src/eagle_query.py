import eagleSqlTools as sql
import numpy as np

def initiate_query(username: str, password: str, simName: str = 'L0100N1504', snapNum: int = 12) -> np.ndarray:
    """Returns a record array containing fof and subhalo properties from EAGLE database.
        Only subhalos that are central to an fof halo are selected.

        Parameters
        ----------
        username: str, required
        Username to access the EAGLE database.

        password: str, required
        Password to access the EAGLE database.

        simName: str, optional
        The name of the DM only EAGLE run.
        Default is L0100N1504.

        snapNum: int, optional
        The snapshot number of the EAGLE run.
        Default is 12 (z ~ 3)

        Returns
        -------
        numpy.ndarray
            Structured array containing subhalo and host halo properties.

        Notes
        -----
        Only central subhalos (SubGroupNumber = 0) with masses between
        1e9 and 1e13 Msun in are selected.
        
        Units:
            - m200: Solar mass
            - r200: Proper kpc
            - x: Comoving Mpc
            - y: Comoving Mpc
            - z: Comoving Mpc
            - vx: km/s
            - vy: km/s
            - vz: km/s 
    """

    if not isinstance(snapNum, int) or isinstance(snapNum, bool):
        raise TypeError("snapNum must be an integer.")

    if not simName.isalnum():
        raise ValueError("simName must contain only letters and numbers.")
    
    con = sql.connect(username, password)
    
    query = f"""
    SELECT
        SH.GalaxyID as ID,
        SH.Redshift as redshift,
        SH.SnapNum as snapnum,
        SH.Mass as mass,
        FoF.Group_M_Crit200 as m200,
        FoF.Group_R_Crit200 as r200,
        SH.CentreOfPotential_x as x,
        SH.CentreOfPotential_y as y,
        SH.CentreOfPotential_z as z,
        SH.Velocity_x as vx,
        SH.Velocity_y as vy,
        SH.Velocity_z as vz
    FROM
        Ref{simName}_Subhalo as SH,
        Ref{simName}_FoF as FoF
    WHERE
        SH.SnapNum = {snapNum}
        and FoF.SnapNum = SH.SnapNum
        and SH.GroupID = FoF.GroupID
        and SH.Mass between 1e9 and 1e13
        and SubGroupNumber = 0
    ORDER BY
    FoF.Group_M_Crit200 DESC
        
    """

    
    halo_table  = sql.execute_query(con, query)
    
    return halo_table