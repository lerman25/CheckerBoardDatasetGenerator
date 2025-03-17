# Synthetic Checkerboard Image Generator

## Overview

This project generates synthetic chessboard images with random camera poses. The script creates a chessboard image, applies a random rotation and translation, and then warps the board so that it remains fully visible within the final image. The generated images are saved along with corresponding intrinsic and extrinsic camera parameters.

## Features

- **Synthetic Chessboard Creation**  
  Generates a black-and-white checkerboard based on user-defined dimensions (number of squares and square size).

- **Random Camera Extrinsics**  
  Samples random roll, pitch, and yaw angles as well as a random translation along the Z-axis to simulate different camera poses.

- **Homography Calculation**  
  Computes the homography that maps the chessboard (assumed to lie on the Z=0 plane) to the image plane using the intrinsic camera parameters.

- **Image Warping and Compositing**  
  Warps the generated chessboard image and composites it onto a green background, ensuring the board is entirely visible.

- **Parameter Logging**  
  - **intrinsics.txt**: Contains the 3×3 intrinsic camera matrix.
  - **extrinsics.csv**: Records each image's extrinsic parameters, including Euler angles, translation vectors, and the flattened rotation matrix.

## Dependencies

- **Python 3.x**
- **OpenCV** (`cv2`)
- **NumPy**

Additional standard libraries: `argparse`, `csv`, `os`, and `random`.

Install the non-standard packages using pip:
```bash
pip install opencv-python numpy
Usage
Run the script using the command line:

bash
Copy
python generate_checkerboard.py [options]
Command-Line Arguments
Argument	Type	Default	Description
--num_images	int	100	Number of images to generate.
--output_dir	str	output_images	Directory to save images and parameter files.
--width	int	1024	Output image width in pixels.
--height	int	768	Output image height in pixels.
--board_rows	int	7	Number of squares vertically on the chessboard.
--board_cols	int	9	Number of squares horizontally on the chessboard.
--square_size	int	30	Size of each square in pixels.
--focal	float	800.0	Focal length in pixels (for the intrinsic camera matrix).
--max_roll	float	5.0	Maximum roll angle (degrees).
--max_pitch	float	15.0	Maximum pitch angle (degrees).
--max_yaw	float	5.0	Maximum yaw angle (degrees).
--min_distance	float	800.0	Minimum translation along Z (closer board).
--max_distance	float	1400.0	Maximum translation along Z (farther board).
Example
bash
Copy
python generate_checkerboard.py \
    --num_images 50 \
    --output_dir my_chessboards \
    --width 800 \
    --height 600 \
    --board_rows 7 \
    --board_cols 9 \
    --square_size 20 \
    --focal 600 \
    --max_roll 10 \
    --max_pitch 20 \
    --max_yaw 10 \
    --min_distance 500 \
    --max_distance 1200
Output Files
Images: Generated images are saved in the specified output directory (e.g., checkerboard_000.png, checkerboard_001.png, etc.).
intrinsics.txt: A text file containing the intrinsic camera matrix.
extrinsics.csv: A CSV file logging the following for each image:
Image filename
Euler angles (roll, pitch, yaw)
Translation vector (tx, ty, tz)
Flattened rotation matrix (9 values)
Code Structure
generate_checkerboard(board_size, square_size)
Creates a grayscale checkerboard image with alternating black and white squares.

eulerAnglesToRotationMatrix(roll, pitch, yaw)
Converts Euler angles (in degrees) to a rotation matrix by applying rotations in the order roll (X-axis), pitch (Y-axis), and yaw (Z-axis).

compute_homography(K, R, t)
Computes the homography mapping points from the chessboard (Z=0) to the image plane using the intrinsic matrix K, rotation matrix R, and translation vector t.

generate_transformed_checkerboard(checkerboard, output_size, H_total)
Warps the chessboard image using the provided homography and composites it onto a green background.

main(args)
Parses command-line arguments, generates multiple images with random extrinsic parameters, saves each image, and writes the intrinsic and extrinsic parameters to their respective files.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests with improvements, additional features, or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for more details.