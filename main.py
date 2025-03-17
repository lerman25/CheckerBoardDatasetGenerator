import cv2
import numpy as np
import random
import os
import argparse
import csv

def generate_checkerboard(board_size, square_size):
    """
    Create a basic chessboard image (black and white pattern).

    Parameters:
      board_size: tuple (rows, cols) representing the number of squares.
      square_size: size in pixels of each square.

    Returns:
      Grayscale numpy array with the chessboard pattern.
    """
    rows, cols = board_size
    board_img = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                board_img[i * square_size:(i + 1) * square_size,
                          j * square_size:(j + 1) * square_size] = 255
    return board_img

def eulerAnglesToRotationMatrix(roll, pitch, yaw):
    """
    Converts Euler angles (in degrees) to a rotation matrix.
    The rotations are applied in the order: roll (X-axis), pitch (Y-axis), then yaw (Z-axis).
    """
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    
    # Rotation about X-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    # Rotation about Y-axis (pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Rotation about Z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return R

def compute_homography(K, R, t):
    """
    Computes the homography mapping points from the chessboard (assumed on the plane Z=0)
    to the image plane.

    For points on Z=0, the projection is given by:
      H = K * [r1, r2, t],
    where r1 and r2 are the first two columns of the rotation matrix R.

    Parameters:
      K: 3x3 intrinsic matrix.
      R: 3x3 rotation matrix.
      t: Translation vector of shape (3,).

    Returns:
      A 3x3 homography matrix.
    """
    H = K @ np.column_stack((R[:, 0], R[:, 1], t))
    return H

def generate_transformed_checkerboard(checkerboard, output_size, H_total):
    """
    Warps the chessboard image using the provided homography and composites it onto a green background.

    Parameters:
      checkerboard: Base chessboard image (grayscale).
      output_size: Tuple (width, height) of the final image.
      H_total: Combined homography matrix (including board centering).

    Returns:
      Final image in BGR color space.
    """
    output_width, output_height = output_size
    # Create a green background (BGR: 0, 255, 0)
    background = np.full((output_height, output_width, 3), (0, 255, 0), dtype=np.uint8)

    # Warp the chessboard using the combined homography.
    warped_checkerboard = cv2.warpPerspective(checkerboard, H_total, (output_width, output_height),
                                                flags=cv2.INTER_CUBIC,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
    # Create a binary mask for the chessboard region.
    board_mask = np.full(checkerboard.shape, 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(board_mask, H_total, (output_width, output_height),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    _, warped_mask_bin = cv2.threshold(warped_mask, 128, 255, cv2.THRESH_BINARY)

    # Convert the warped chessboard image to BGR.
    warped_color = cv2.cvtColor(warped_checkerboard, cv2.COLOR_GRAY2BGR)

    # Composite the warped chessboard over the green background using the mask.
    composite = background.copy()
    composite[warped_mask_bin == 255] = warped_color[warped_mask_bin == 255]

    return composite

def main(args):
    # Create output directory if it doesn't exist.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up the intrinsic camera parameters.
    cx = args.width / 2
    cy = args.height / 2
    K = np.array([[args.focal, 0, cx],
                  [0, args.focal, cy],
                  [0, 0, 1]])
    
    # Save the intrinsic parameters.
    intrinsic_file = os.path.join(args.output_dir, "intrinsics.txt")
    with open(intrinsic_file, "w") as f:
        f.write("Intrinsic Camera Matrix K:\n")
        f.write(np.array2string(K, separator=", ") + "\n")
    print(f"Saved intrinsic parameters to {intrinsic_file}")
    
    # Create the base chessboard image.
    checkerboard = generate_checkerboard((args.board_rows, args.board_cols), args.square_size)
    board_h, board_w = checkerboard.shape
    
    # Precompute the translation matrix T that centers the board.
    T = np.array([[1, 0, -board_w / 2],
                  [0, 1, -board_h / 2],
                  [0, 0, 1]])
    
    # Open a CSV file to record the extrinsic parameters for each image.
    extrinsics_csv = os.path.join(args.output_dir, "extrinsics.csv")
    with open(extrinsics_csv, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # CSV header: image filename, Euler angles (roll, pitch, yaw), translations, and flattened rotation matrix.
        header = ["image", "roll", "pitch", "yaw", "tx", "ty", "tz"] + [f"R{i}" for i in range(9)]
        csv_writer.writerow(header)
        
        for i in range(args.num_images):
            # Random rotations.
            roll = random.uniform(-args.max_roll, args.max_roll)
            yaw = random.uniform(-args.max_yaw, args.max_yaw)
            pitch = random.uniform(-args.max_pitch, args.max_pitch)
            # Random board distance along Z.
            tz = random.uniform(args.min_distance, args.max_distance)
            
            # Compute rotation matrix.
            R = eulerAnglesToRotationMatrix(roll, pitch, yaw)
            
            # Now, we must choose a board-center position (x_target, y_target)
            # such that the entire projected board is within the image.
            max_attempts = 100
            valid = False
            for attempt in range(max_attempts):
                # Sample a random desired board-center position.
                x_target = random.uniform(0, args.width)
                y_target = random.uniform(0, args.height)
                # Compute tx, ty so that the board center projects to (x_target, y_target)
                tx = (x_target - cx) * tz / args.focal
                ty = (y_target - cy) * tz / args.focal
                t = np.array([tx, ty, tz])
                H = compute_homography(K, R, t)
                H_total = H @ T
                # Project the four corners of the board.
                corners = np.array([
                    [-board_w/2, -board_h/2, 1],
                    [ board_w/2, -board_h/2, 1],
                    [ board_w/2,  board_h/2, 1],
                    [-board_w/2,  board_h/2, 1]
                ]).T  # shape 3x4
                proj = H_total @ corners  # shape 3x4
                proj /= proj[2, :]  # Normalize homogeneous coordinates.
                xs = proj[0, :]
                ys = proj[1, :]
                # Check if all corners are inside the image.
                if xs.min() >= 0 and xs.max() <= args.width and ys.min() >= 0 and ys.max() <= args.height:
                    valid = True
                    break
            
            if not valid:
                print(f"Warning: Could not find a valid board center for image {i} after {max_attempts} attempts. Using last computed values.")
            
            # Use the found values.
            t = np.array([tx, ty, tz])
            H = compute_homography(K, R, t)
            H_total = H @ T
            
            # Generate the transformed image.
            img = generate_transformed_checkerboard(checkerboard, (args.width, args.height), H_total)
            image_filename = os.path.join(args.output_dir, f"checkerboard_{i:03d}.png")
            cv2.imwrite(image_filename, img)
            print(f"Saved {image_filename} with roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}, t=({tx:.2f}, {ty:.2f}, {tz:.2f}), board center=({x_target:.1f}, {y_target:.1f})")
            
            # Flatten rotation matrix for saving.
            R_flat = R.flatten().tolist()
            csv_writer.writerow([f"checkerboard_{i:03d}.png", roll, pitch, yaw, tx, ty, tz] + R_flat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate n synthetic chessboard images with random camera extrinsics (R, t), varying board positions and distances, ensuring the board is fully visible, then save corresponding parameters."
    )
    parser.add_argument('--num_images', type=int, default=100, help="Number of images to generate")
    parser.add_argument('--output_dir', type=str, default='output_images', help="Directory to save images and parameters")
    parser.add_argument('--width', type=int, default=1024, help="Output image width in pixels")
    parser.add_argument('--height', type=int, default=768, help="Output image height in pixels")
    parser.add_argument('--board_rows', type=int, default=7, help="Number of squares vertically on the chessboard")
    parser.add_argument('--board_cols', type=int, default=9, help="Number of squares horizontally on the chessboard")
    parser.add_argument('--square_size', type=int, default=30, help="Size of each square in pixels")
    parser.add_argument('--focal', type=float, default=800.0, help="Focal length (in pixels)")
    
    # Ranges for random extrinsics.
    parser.add_argument('--max_roll', type=float, default=5.0, help="Maximum roll angle (degrees)")
    parser.add_argument('--max_pitch', type=float, default=15.0, help="Maximum pitch angle (degrees) for forward/backward tilt")
    parser.add_argument('--max_yaw', type=float, default=5.0, help="Maximum yaw angle (degrees)")
    # Use a distance range for Z.
    parser.add_argument('--min_distance', type=float, default=800.0, help="Minimum translation along Z (closer board)")
    parser.add_argument('--max_distance', type=float, default=1400.0, help="Maximum translation along Z (farther board)")
    
    args = parser.parse_args()
    main(args)
