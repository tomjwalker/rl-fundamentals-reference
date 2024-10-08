import matplotlib
import numpy as np
import argparse
import sys

# Use TkAgg backend for matplotlib
from exercises.mdps.utils import display_values, generate_videos, stitch_videos

matplotlib.use('TkAgg')

# Discount factor
GAMMA = 0.5

# Predefined maps
MAP_TWO_TWO = [
    "SF",
    "FG"
]

MAP_THREE_THREE = [
    "SFF",
    "HFH",
    "HFG"
]


def solve_two_by_two():
    """
    Solves the 2x2 FrozenLake map by setting up and solving the linear system Av = b.

    Returns:
        v (numpy.ndarray): Array of state values.
    """
    # ASSIGNMENT START: Work out the simultaneous equations (via the Bellman equation) for the 2x2 FrozenLake map
    # for the equiprobable policy and a deterministic environment (we went through this in the lecture, for reference).
    # Write down in matrix form, Av = b, where A is the transition matrix, v is the state values,
    # and b is the reward vector. Fill in the elements of the A and b array below (replace the None values).
    # Write elements as a function of the discount factor GAMMA.
    A = np.array([
        [None, None, None],
        [None, None, None],
        [None, None, None],
    ])

    # Define the reward vector
    b = np.array([None, None, None])
    # ASSIGNMENT END

    # SOLUTION START:
    A = np.array([
        [(1 - GAMMA / 2), (-GAMMA / 4), (-GAMMA / 4)],
        [(-GAMMA / 4), (1 - GAMMA / 2), 0],
        [(-GAMMA / 4), 0, (1 - GAMMA / 2)],
    ])

    # Define the reward vector
    b = np.array([0, 1 / 4, 1 / 4])
    # SOLUTION END

    # Solve the linear system Av = b
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Failed to solve the linear system for 2x2 map: {e}")
        v = None

    return v


def solve_three_by_three():
    """
    Solves the 3x3 FrozenLake map by setting up and solving the linear system Av = b.

    Returns:
        v (numpy.ndarray): Array of state values.
    """

    # ASSIGNMENT START:
    # Work out the transition matrix A for the 3x3 map provided (note the holes).
    # Only states 0, 1, 2, 4, and 7 are non-terminal, so the matrix is 5x5.
    # Replace the None values with the correct expressions in terms of the discount factor GAMMA.
    A = np.array([
        [None, None, None, None, None],
        [None, None, None, None, None],
        [None, None, None, None, None],
        [None, None, None, None, None],
        [None, None, None, None, None],
    ])

    # Define the reward vector
    b = np.array([None, None, None, None, None])
    # ASSIGNMENT END

    # SOLUTION START:
    # Define the transition matrix
    # Only states 0, 1, 2, 4, and 7 are non-terminal, so the matrix is 5x5
    A = np.array([
        [(1 - GAMMA / 2), (-GAMMA / 4), 0, 0, 0],
        [(-GAMMA / 4), (1 - GAMMA / 4), (-GAMMA / 4), (-GAMMA / 4), 0],
        [0, (-GAMMA / 4), (1 - GAMMA / 2), 0, 0],
        [0, (-GAMMA / 4), 0, 1, (-GAMMA / 4)],
        [0, 0, 0, (-GAMMA / 4), (1 - GAMMA / 4)],
    ])

    # Define the reward vector
    b = np.array([0, 0, 0, 0, 1 / 4])
    # SOLUTION END

    # Solve the linear system
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Failed to solve the linear system for 3x3 map: {e}")
        v = None

    return v


def main():
    """
    Main function to handle command-line arguments and execute corresponding actions.
    """
    parser = argparse.ArgumentParser(description="FrozenLake Analytical Investigations")
    subparsers = parser.add_subparsers(dest='mode', help='Modes of operation')

    # Subparser for 'solve' mode
    parser_solve = subparsers.add_parser('solve', help='Solve specific 2x2 and 3x3 FrozenLake maps')

    # Subparser for 'video' mode
    parser_video = subparsers.add_parser('video', help='Generate videos of equiprobable agent episodes')
    parser_video.add_argument('--episodes', type=int, default=5, help='Number of episodes to record per map (default: 5)')
    parser_video.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos (default: videos)')

    # Subparser for 'stitch' mode
    parser_stitch = subparsers.add_parser('stitch', help='Stitch episode videos into a single video')
    parser_stitch.add_argument('--map_name', type=str, required=True, help='Name of the map to stitch (e.g., FrozenLake_2x2)')
    parser_stitch.add_argument('--video_dir', type=str, default='videos', help='Directory where episode videos are stored (default: videos)')
    parser_stitch.add_argument('--output_dir', type=str, default='stitched_videos', help='Directory to save stitched video (default: stitched_videos)')
    parser_stitch.add_argument('--loop', action='store_true', help='Whether to loop the stitched video')
    parser_stitch.add_argument('--loop_count', type=int, default=1, help='Number of times to loop the video if loop=True (default: 1)')

    args = parser.parse_args()

    if args.mode == 'solve':
        # Solve and display the two by two grid
        print("Solving 2x2 Frozen Lake:")
        v_two_by_two = solve_two_by_two()
        display_values(v_two_by_two, MAP_TWO_TWO)

        # Solve and display the three by three grid
        print("Solving 3x3 Frozen Lake:")
        v_three_by_three = solve_three_by_three()
        display_values(v_three_by_three, MAP_THREE_THREE)

    elif args.mode == 'video':
        # Generate videos for 2x2 map
        print("Generating videos for 2x2 Frozen Lake:")
        generate_videos(desc=MAP_TWO_TWO,
                        map_name='FrozenLake_2x2',
                        num_episodes=args.episodes,
                        video_dir=args.video_dir)

        # Generate videos for 3x3 map
        print("Generating videos for 3x3 Frozen Lake:")
        generate_videos(desc=MAP_THREE_THREE,
                        map_name='FrozenLake_3x3',
                        num_episodes=args.episodes,
                        video_dir=args.video_dir)

    elif args.mode == 'stitch':
        # Stitch videos for the specified map
        print(f"Stitching videos for map '{args.map_name}':")
        stitch_videos(map_name=args.map_name,
                      video_dir=args.video_dir,
                      output_dir=args.output_dir,
                      loop=args.loop,
                      loop_count=args.loop_count)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
