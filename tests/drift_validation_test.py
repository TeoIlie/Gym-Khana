#!/usr/bin/env python3
"""
Drift Validation Test Script for F1TENTH Gym

This script validates that the F1TENTH simulation can express lateral drift behavior.
It demonstrates a complete drift maneuver sequence:
1. Accelerate to high speed
2. Execute abrupt steering to initiate drift
3. Monitor lateral velocity (vy) to confirm drift physics

Expected behavior: vy should transition from ~0 to positive values during drift.
"""

import gymnasium as gym
import numpy as np
import time


def print_observation_info(obs, step, phase_name=""):
    """Print formatted observation values showing drift state"""
    vx, vy, yaw_rate, delta, frenet_u, frenet_n = obs

    print(
        f"Step {step:4d} {phase_name:12s}: "
        f"vx={vx:6.2f}, vy={vy:6.2f}, yaw_rate={yaw_rate:6.2f}, "
        f"delta={delta:6.3f}, frenet_u={frenet_u:6.2f}, frenet_n={frenet_n:6.2f}"
    )


def main():
    print("=" * 80)
    print("F1TENTH Drift Validation Test")
    print("=" * 80)
    print("This test demonstrates lateral drift physics in the F1TENTH simulation.")
    print("Expected sequence:")
    print("1. Accelerate straight to high speed (>15 m/s)")
    print("2. Execute sharp steering input to initiate drift")
    print("3. Monitor vy (lateral velocity) transition from ~0 to positive values")
    print("=" * 80)

    # Environment configuration for drift testing
    config = {
        "map": "Spielberg_blank",  # Empty map for clear drift demonstration
        "num_agents": 1,
        "timestep": 0.01,  # 100Hz for accurate physics
        "integrator": "rk4",  # Accurate physics integration
        "model": "st",  # Single track model with tire slip for drift
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "drift"},  # 6D drift state: [vx, vy, yaw_rate, delta, frenet_u, frenet_n]
        "reset_config": {"type": "rl_random_static"},
    }

    # Create environment with rendering
    try:
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=config,
            render_mode="human",
        )
        print("✅ Environment created successfully with rendering")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print_observation_info(obs, 0, "INITIAL")

    # Try initial rendering
    try:
        env.render()
        print("✅ Rendering working - you should see the track window")
        time.sleep(1)  # Give time to see initial state
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        print("Continuing without visualization...")

    # Test phases
    drift_detected = False
    max_vy = 0.0
    max_speed = 0.0

    print("\n" + "=" * 60)
    print("PHASE 1: ACCELERATION - Building up speed")
    print("=" * 60)

    # Phase 1: Accelerate straight to high speed
    acceleration_steps = 300  # ~3 seconds at 100Hz
    target_speed = 18.0  # m/s - high speed for drift initiation

    for step in range(1, acceleration_steps + 1):
        # Straight acceleration: no steering, high speed command
        action = np.array([[0.0, target_speed]])  # [steering_angle, speed]

        obs, reward, done, truncated, info = env.step(action)
        vx, vy, yaw_rate, delta, frenet_u, frenet_n = obs

        max_speed = max(max_speed, abs(vx))

        # Print every 50 steps during acceleration
        if step % 50 == 0 or step <= 5:
            print_observation_info(obs, step, "ACCEL")

        # Render
        try:
            env.render()
        except Exception:
            pass

        # Check if we've reached good speed
        if abs(vx) > 15.0 and step > 100:
            print(f"✅ Target speed reached: {abs(vx):.2f} m/s at step {step}")
            break

        if done or truncated:
            print(f"❌ Episode ended early during acceleration at step {step}")
            break

    print(f"Maximum speed achieved: {max_speed:.2f} m/s")

    print("\n" + "=" * 60)
    print("PHASE 2: DRIFT INITIATION - Sharp steering input")
    print("=" * 60)

    # Phase 2: Execute sharp steering to initiate drift
    drift_steps = 200  # ~2 seconds for drift maneuver
    drift_steering = 0.35  # Significant steering angle in radians (~20 degrees)
    drift_speed = 15.0  # Maintain high speed during drift

    for step in range(acceleration_steps + 1, acceleration_steps + drift_steps + 1):
        # Sharp steering while maintaining speed
        action = np.array([[drift_steering, drift_speed]])

        obs, reward, done, truncated, info = env.step(action)
        vx, vy, yaw_rate, delta, frenet_u, frenet_n = obs

        max_vy = max(max_vy, abs(vy))

        # Print every step initially, then every 10 steps
        relative_step = step - acceleration_steps
        if relative_step <= 10 or relative_step % 10 == 0:
            print_observation_info(obs, step, "DRIFT")

        # Check for drift detection (significant lateral velocity)
        if abs(vy) > 2.0 and not drift_detected:
            drift_detected = True
            print(f"🎯 DRIFT DETECTED! Lateral velocity vy = {vy:.3f} m/s at step {step}")
            print("   Vehicle is now sliding laterally!")

        # Render
        try:
            env.render()
        except Exception:
            pass

        if done or truncated:
            print(f"Episode ended during drift at step {step}")
            break

    print("\n" + "=" * 60)
    print("PHASE 3: DRIFT MONITORING - Sustained lateral motion")
    print("=" * 60)

    # Phase 3: Continue drift monitoring
    monitor_steps = 150  # Additional monitoring

    for step in range(acceleration_steps + drift_steps + 1, acceleration_steps + drift_steps + monitor_steps + 1):
        # Continue with reduced steering to observe drift dynamics
        action = np.array([[drift_steering * 0.7, drift_speed * 0.9]])

        obs, reward, done, truncated, info = env.step(action)
        vx, vy, yaw_rate, delta, frenet_u, frenet_n = obs

        max_vy = max(max_vy, abs(vy))

        # Print every 20 steps
        relative_step = step - acceleration_steps - drift_steps
        if relative_step % 20 == 0 or relative_step <= 5:
            print_observation_info(obs, step, "MONITOR")

        # Render
        try:
            env.render()
        except Exception:
            pass

        if done or truncated:
            print(f"Episode ended during monitoring at step {step}")
            break

    # Final results summary
    print("\n" + "=" * 80)
    print("DRIFT VALIDATION RESULTS")
    print("=" * 80)

    print(f"Maximum longitudinal velocity (vx): {max_speed:.2f} m/s")
    print(f"Maximum lateral velocity (vy):      {max_vy:.2f} m/s")
    print(f"Drift behavior detected:            {'✅ YES' if drift_detected else '❌ NO'}")

    # Success criteria evaluation
    speed_achieved = max_speed > 15.0
    lateral_motion_detected = max_vy > 1.5

    print("\nSuccess Criteria:")
    print(f"• High speed achieved (>15 m/s):     {'✅ PASS' if speed_achieved else '❌ FAIL'}")
    print(f"• Lateral drift detected (vy >1.5):  {'✅ PASS' if lateral_motion_detected else '❌ FAIL'}")

    overall_success = speed_achieved and lateral_motion_detected and drift_detected
    print(f"• Overall drift validation:          {'✅ PASS' if overall_success else '❌ FAIL'}")

    if overall_success:
        print("\n🎉 SUCCESS: F1TENTH simulation successfully demonstrates drift physics!")
        print("   The lateral velocity (vy) showed clear positive values during the drift maneuver,")
        print("   confirming that the simulation can model realistic vehicle sliding behavior.")
    else:
        print("\n⚠️  WARNING: Drift behavior may not be fully demonstrated.")
        print("   Consider adjusting steering angle, speed, or surface friction parameters.")

    print("=" * 80)

    # Keep window open briefly for final observation
    try:
        time.sleep(2)
        env.render()
    except Exception:
        pass

    env.close()
    print("Test completed. Environment closed.")


if __name__ == "__main__":
    main()
