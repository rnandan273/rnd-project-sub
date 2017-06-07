import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    print(Rover.mode)
    ###

    if Rover.nav_angles is not None:
        if Rover.picking_up == 1:
            Rover.throttle = 0
            Rover.steer = 0
            Rover.brake = Rover.brake_set
            Rover.mode = 'stop'
        elif Rover.near_sample == 1:
                Rover.brake = Rover.brake_set
                Rover.mode = 'stop'
                Rover.steer = 0
                Rover.send_pickup = True
                Rover.samples_found += 1

        elif Rover.mode == 'stop':
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            else:
                Rover.throttle = Rover.throttle_set
                Rover.steer = 0
                Rover.brake = 0
                if Rover.steer == 0:
                    Rover.steer = -15 if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0 else 15
                Rover.mode = 'forward'
        elif Rover.vel <= 0.2:
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    Rover.brake = 0
                    if Rover.steer == 0:
                        Rover.steer = -15 if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0 else 15
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle of mean angle of terrain angles and mean angle of wall angles
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

        elif Rover.mode == 'rock_visible':
                print(Rover.max_vel)
                print(Rover.vel)
                if Rover.vel < Rover.max_vel:
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
        elif Rover.mode == 'forward':
                if Rover.vel < Rover.max_vel:
                    Rover.throttle = Rover.throttle_set
                    Rover.brake = 0
                    # Set steering to average angle clipped to the range +/- 15
                    Rover.steer =  -15 if np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15) < 0 else 15
                else:
                    Rover.throttle = 0
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
  ###
    return Rover
