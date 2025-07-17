import cv2
import numpy as np
import random
import os
import argparse
import csv

def generate_checkerboard(rows, cols, square_size, border_width=2):
    # Create image with border
    total_width = cols * square_size + 2 * border_width * square_size
    total_height = rows * square_size + 2 * border_width * square_size
    img = np.ones((total_height, total_width), dtype=np.uint8) * 255

    # Generate checkerboard pattern in the center
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                y1 = border_width * square_size + i * square_size
                y2 = border_width * square_size + (i + 1) * square_size
                x1 = border_width * square_size + j * square_size
                x2 = border_width * square_size + (j + 1) * square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)

    offset_x = border_width * square_size
    offset_y = border_width * square_size
    inner_w = cols * square_size
    inner_h = rows * square_size
    return img, offset_x, offset_y, inner_w, inner_h


def euler_to_rotation_matrix(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def compute_homography(K, R, t):
    return K @ np.hstack((R[:, :2], t.reshape(3, 1)))

def warp_and_composite(checkerboard, offset_x, offset_y, H, img_size, noise_stddev=1.0):
    """
    Warps the checkerboard using a homography directly onto a gray background.
    No masking or compositing required. Preserves sharp black/white contrast.
    """
    h, w = img_size
    # Start with gray background
    canvas = np.full((h, w), 128, dtype=np.uint8)

    # Centering transform
    T = np.array([[1, 0, -offset_x],
                  [0, 1, -offset_y],
                  [0, 0, 1]])
    H_total = H @ T

    # Warp the checkerboard directly into the gray canvas using nearest-neighbor
    warped = cv2.warpPerspective(checkerboard, H_total, (w, h),
                                 dst=canvas,              # <-- warp into canvas directly!
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_TRANSPARENT)

    # Convert to BGR
    composite = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Optional: add mild noise
    if noise_stddev > 0:
        noise = np.random.normal(0, noise_stddev, composite.shape).astype(np.int16)
        composite = np.clip(composite.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return composite




def generate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    K = np.array([
        [args.focal, 0, args.width / 2],
        [0, args.focal, args.height / 2],
        [0, 0, 1]
    ])
    np.savetxt(os.path.join(args.output_dir, "intrinsics.txt"), K, fmt="%.4f")

    board_img, offset_x, offset_y, inner_w, inner_h = generate_checkerboard(
        args.board_rows, args.board_cols, args.square_size
    )
    board_h, board_w = board_img.shape
    corners = np.array([
        [0, 0, 1],
        [inner_w, 0, 1],
        [inner_w, inner_h, 1],
        [0, inner_h, 1]
    ]).T
    csv_path = os.path.join(args.output_dir, "extrinsics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "roll", "pitch", "yaw", "tx", "ty", "tz"] + [f"R{i}" for i in range(9)])

        for idx in range(args.num_images):
            for attempt in range(50):
                roll = random.uniform(-args.max_roll, args.max_roll)
                pitch = random.uniform(-args.max_pitch, args.max_pitch)
                yaw = random.uniform(-args.max_yaw, args.max_yaw)
                tz = random.uniform(args.min_distance, args.max_distance)

                R = euler_to_rotation_matrix(roll, pitch, yaw)

                x_img = random.uniform(0.3, 0.7) * args.width
                y_img = random.uniform(0.3, 0.7) * args.height

                tx = (x_img - args.width / 2) * tz / args.focal
                ty = (y_img - args.height / 2) * tz / args.focal
                t = np.array([tx, ty, tz])

                H = compute_homography(K, R, t)
                proj = H @ np.vstack((corners[0] - offset_x, corners[1] - offset_y, corners[2]))
                proj /= proj[2]

                if (proj[0] >= 0).all() and (proj[0] < args.width).all() and \
                   (proj[1] >= 0).all() and (proj[1] < args.height).all():
                    break
            else:
                print(f"Failed to find a valid projection for image {idx}")
                continue

            image = warp_and_composite(board_img, offset_x, offset_y, H, (args.height, args.width))
            filename = f"checkerboard_{idx:03d}.png"
            cv2.imwrite(os.path.join(args.output_dir, filename), image)

            writer.writerow([filename, roll, pitch, yaw, tx, ty, tz] + R.flatten().tolist())

            print(f"[{idx+1}/{args.num_images}] Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic synthetic chessboard images for camera calibration.")
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='output_images')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--board_rows', type=int, default=8)
    parser.add_argument('--board_cols', type=int, default=10)
    parser.add_argument('--square_size', type=int, default=25)
    parser.add_argument('--focal', type=float, default=1000.0)
    parser.add_argument('--max_roll', type=float, default=5.0)
    parser.add_argument('--max_pitch', type=float, default=10.0)
    parser.add_argument('--max_yaw', type=float, default=5.0)
    parser.add_argument('--min_distance', type=float, default=800.0)
    parser.add_argument('--max_distance', type=float, default=1300.0)

    args = parser.parse_args()
    generate_dataset(args)
