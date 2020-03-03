try:
    from scipy import constants
    from scipy import pi

    q = constants.elementary_charge
    e = constants.elementary_charge
    k = constants.k
    k_eV = k/q
    h = constants.h
    hbar = constants.hbar
    eps0 = constants.epsilon_0
    coulomb_const = (q / (4*pi*eps0))
    
except:
    from scipy import pi

    q = 1.60217653e-19
    e = 1.60217653e-19
    k = 1.3806505000000001e-23
    k_eV = k/q
    h = 6.6260693000000002e-34
    hbar = 1.0545716823644548e-34
    eps0 = 8.8541878176203892e-12
    coulomb_const = (q / (4*pi*eps0))
