import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160) , rgb_thresh_max=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2]) \
                &(img[:,:,0] < rgb_thresh_max[0]) \
                & (img[:,:,1] < rgb_thresh_max[1]) \
                & (img[:,:,2] < rgb_thresh_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 255
    #color_select[below_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    img = Rover.img
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    #plt.imshow(warped)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    wall_threshed = color_thresh(warped, rgb_thresh=(0, 0, 0), rgb_thresh_max=(160, 160, 160))
    rock_threshed = color_thresh(warped, rgb_thresh=(150, 100, 0), rgb_thresh_max=(255, 200, 50))
    # 4) Convert thresholded image pixel values to rover-centric coords

    Rover.vision_image[:, :, 2] = rock_threshed # red
    Rover.vision_image[:, :, 0] = wall_threshed # green
    Rover.vision_image[:, :, 1] = threshed # blue

    xpix_nav, ypix_nav = rover_coords(threshed)
    xpix_wall, ypix_wall = rover_coords(wall_threshed)
    xpix_rock, ypix_rock = rover_coords(rock_threshed)
    # 5) Convert rover-centric pixel values to world coords
    scale = 10

    xpix_world, ypix_world = pix_to_world(xpix_nav, ypix_nav, Rover.pos[0],
                                Rover.pos[1], Rover.yaw,
                                Rover.worldmap.shape[0], scale)

    xpix_world_wall, ypix_world_wall = pix_to_world(xpix_wall, ypix_wall, Rover.pos[0],
                                Rover.pos[1], Rover.yaw,
                                Rover.worldmap.shape[0], scale)
    xpix_world_rock, ypix_world_rock = pix_to_world(xpix_rock, ypix_rock, Rover.pos[0],
                                Rover.pos[1], Rover.yaw,
                                Rover.worldmap.shape[0], scale)

    Rover.worldmap[ypix_world, xpix_world, 2] += 1
    Rover.worldmap[ypix_world, xpix_world, 0] -= 1

    Rover.worldmap[ypix_world_wall, xpix_world_wall, 2] -= 0
    Rover.worldmap[ypix_world_rock, xpix_world_rock, 0] += 1


    rock_dist, rock_angles = to_polar_coords(xpix_rock, ypix_rock)
    #print("perception")
    #print(rock_dist)
    #print(rock_angles)

    # Does the rover see a rock
    if len(rock_dist) > 0:
        print(len(rock_dist))

        Rover.mode = 'rock_visible'
        Rover.nav_dists = rock_dist
        Rover.nav_angles = rock_angles
        #print('ROVER SEE ROCK')
        #print('steering angle for ROCK = %f' % (np.clip(np.mean(rock_angles * 180/np.pi), -15, 15)))
    else:

        if Rover.mode == 'rock_visible':
            Rover.mode = 'forward'

        dist, angles = to_polar_coords(xpix_nav, ypix_nav)
        Rover.nav_dists = dist
        Rover.nav_angles = angles

    #print("Before returning from perception", Rover.mode)
    return Rover
