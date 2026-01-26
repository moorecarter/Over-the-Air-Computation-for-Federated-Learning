"""
USRP utilities with simulation support
"""

import os

# Check if we're in simulation mode
SIMULATION_MODE = os.environ.get('FL_OTA_SIMULATION', 'False').lower() == 'true'

if SIMULATION_MODE:
    print("=" * 60)
    print("RUNNING IN SIMULATION MODE - No USRP hardware required")
    print("=" * 60)
    from .usrp_simulator import USRP_X310_Simulator as USRP_X310
    from .usrp_simulator import csi_estimate_simulator as csi_estimate
else:
    try:
        from .usrp_x310 import USRP_X310
        from .csi_estimate import csi_estimate
    except ImportError as e:
        print(f"Warning: Could not import USRP modules: {e}")
        print("Falling back to simulation mode...")
        from .usrp_simulator import USRP_X310_Simulator as USRP_X310
        from .usrp_simulator import csi_estimate_simulator as csi_estimate