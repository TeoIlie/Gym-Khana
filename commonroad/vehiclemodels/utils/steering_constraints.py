def steering_constraints(steering_angle, steering_velocity, p):
    """
    steering_constraints - adjusts the steering velocity based on steering

    Inputs:
        :param steering_angle - steering angle
        :param steering_velocity - steering velocity
        :params p - steering parameter structure

    Outputs:
        :return steering_velocity - steering velocity

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
    # steering limit reached?
    if (steering_angle <= p.min and steering_velocity <= 0) or (steering_angle >= p.max and steering_velocity >= 0):
        steering_velocity = 0
    elif steering_velocity <= p.v_min:
        steering_velocity = p.v_min
    elif steering_velocity >= p.v_max:
        steering_velocity = p.v_max

    return steering_velocity


def kappa_dot_dot_constraints(kappa_dot_dot, kappa_dot, p):
    """
    input constraints for kappa_dot_dot: adjusts kappa_dot_dot if kappa_dot limit (i.e., maximum curvature rate)
    or input bounds are reached
    """
    if (kappa_dot < -p.kappa_dot_max and kappa_dot_dot < 0.0) or (kappa_dot > p.kappa_dot_max and kappa_dot_dot > 0.0):
        # kappa_dot limit reached
        kappa_dot_dot = 0.0
    elif abs(kappa_dot_dot) >= p.kappa_dot_dot_max:
        # input bounds reached
        kappa_dot_dot = p.kappa_dot_dot_max
    return kappa_dot_dot
