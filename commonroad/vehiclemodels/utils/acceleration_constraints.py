def acceleration_constraints(velocity, acceleration, p):
    """
    accelerationConstraints - adjusts the acceleration based on acceleration constraints

    Inputs:
        :param acceleration - acceleration in driving direction
        :param velocity - velocity in driving direction
        :params p - longitudinal parameter structure

    Outputs:
        :return acceleration - acceleration in driving direction

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
    # positive acceleration limit
    if velocity > p.v_switch:
        posLimit = p.a_max * p.v_switch / velocity
    else:
        posLimit = p.a_max

    # acceleration limit reached?
    if (velocity <= p.v_min and acceleration <= 0) or (velocity >= p.v_max and acceleration >= 0):
        acceleration = 0
    elif acceleration <= -p.a_max:
        acceleration = -p.a_max
    elif acceleration >= posLimit:
        acceleration = posLimit

    return acceleration


def jerk_dot_constraints(jerk_dot, jerk, p):
    """
    input constraints for jerk_dot: adjusts jerk_dot if jerk limit or input bounds are reached
    """
    if (jerk_dot < 0.0 and jerk <= -p.j_max) or (jerk_dot > 0.0 and jerk >= p.j_max):
        # jerk limit reached
        jerk_dot = 0.0
    elif abs(jerk_dot) >= p.j_dot_max:
        # input bounds reached
        jerk_dot = p.j_dot_max
    return jerk_dot
