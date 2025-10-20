from omegaconf import DictConfig
from commonroad.vehiclemodels.vehicle_parameters import setup_vehicle_parameters, VehicleParameters


def parameters_vehicle1() -> VehicleParameters:
    """
    Creates a VehicleParameters object holding all vehicle parameters vehicle ID 1 (Ford Escort)
    """
    return setup_vehicle_parameters(vehicle_id=1)


# Test parameters
if __name__ == "__main__":
    params = parameters_vehicle1()
