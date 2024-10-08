import os

import gymnasium as gym
import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

# Define actions
ACTIONS = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right


def display_values(v, desc):
    """
    Displays the state values in a grid format corresponding to the map.

    Args:
        v (numpy.ndarray): Array of state values for non-terminal states.
        desc (list of str): Map description.
    """
    if v is None:
        print("No values to display due to previous errors.\n")
        return

    grid = [list(row) for row in desc]
    rows = len(grid)
    cols = len(grid[0])
    num_states = rows * cols

    # Identify terminal states and their rewards
    terminal_rewards = {}
    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            if grid[r][c] == 'H':
                terminal_rewards[state] = 0  # Hole has reward 0
            elif grid[r][c] == 'G':
                terminal_rewards[state] = 0  # Goal has reward 1

    # Initialize a full state value array
    full_v = np.zeros(num_states)

    # Assign rewards to terminal states
    for state, reward in terminal_rewards.items():
        full_v[state] = reward

    # Assign values from v to non-terminal states
    v_index = 0
    for state in range(num_states):
        if state not in terminal_rewards:
            if v is not None and v_index < len(v):
                full_v[state] = v[v_index]
                v_index += 1
            else:
                full_v[state] = 0  # Default value if v is incomplete

    # Display the grid with state values
    print("State Values:")
    for r in range(rows):
        row_vals = ""
        for c in range(cols):
            state = r * cols + c
            row_vals += f"{full_v[state]:.3f} "
        print(row_vals)
    print("\n")



def generate_videos(desc, map_name, num_episodes=5, video_dir='videos'):
    """
    Generates videos of episodes using an equiprobable agent for the specified map.

    Args:
        desc (list of str): Map description.
        map_name (str): Name identifier for the map.
        num_episodes (int): Number of episodes to record.
        video_dir (str): Directory to save videos.
    """
    # Create the video directory if it doesn't exist
    map_video_dir = os.path.join(video_dir, f"{map_name}_videos")
    os.makedirs(map_video_dir, exist_ok=True)
    print(f"Recording videos to: {map_video_dir}")

    # Create the FrozenLake environment with 'rgb_array' render mode
    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   render_mode='rgb_array',
                   is_slippery=False)

    # Wrap the environment to record videos
    env = gym.wrappers.RecordVideo(env,
                                   video_folder=map_video_dir,
                                   episode_trigger=lambda episode_id: episode_id < num_episodes,
                                   name_prefix=map_name)

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_over = False
        step = 0
        while not episode_over:
            action = np.random.choice(len(ACTIONS))  # Equiprobable agent
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            # Optionally render each step (commented out to speed up)
            # frame = env.render()
            # if frame is not None:
            #     print(f"Frame captured at step {step}")
            episode_over = terminated or truncated
        print(f"Episode {episode + 1} completed in {step} steps.")

    # Properly close the environment to ensure videos are saved
    env.close()
    print(f"Videos saved to {map_video_dir}\n")


def stitch_videos(map_name, video_dir='videos', output_dir='stitched_videos', loop=False, loop_count=1):
    """
    Stitches all episode videos for a particular map into a single video, optionally looping.

    Args:
        map_name (str): Name identifier for the map (e.g., 'FrozenLake_2x2').
        video_dir (str): Directory where episode videos are stored.
        output_dir (str): Directory to save the stitched video.
        loop (bool): Whether to loop the video.
        loop_count (int): Number of times to loop the video if loop=True.

    Returns:
        None
    """
    # Path to the map's video directory
    map_video_dir = os.path.join(video_dir, f"{map_name}_videos")

    if not os.path.exists(map_video_dir):
        print(f"Error: Video directory '{map_video_dir}' does not exist.")
        return

    # Get list of video files sorted by name
    video_files = [f for f in os.listdir(map_video_dir) if f.endswith('.mp4')]
    video_files.sort()  # Ensure videos are in order

    if not video_files:
        print(f"Error: No video files found in '{map_video_dir}'.")
        return

    # Load video clips
    clips = []
    for video_file in video_files:
        video_path = os.path.join(map_video_dir, video_file)
        try:
            clip = VideoFileClip(video_path)
            clips.append(clip)
            print(f"Loaded video: {video_path}")
        except Exception as e:
            print(f"Warning: Failed to load video '{video_path}': {e}")

    if not clips:
        print("Error: No clips loaded. Exiting stitching process.")
        return

    # Concatenate clips
    try:
        final_clip = concatenate_videoclips(clips, method='compose')
        print("Successfully concatenated all clips.")
    except Exception as e:
        print(f"Error: Failed to concatenate clips: {e}")
        return

    # Loop the video if required
    if loop and loop_count > 1:
        final_clip = concatenate_videoclips([final_clip] * loop_count)
        print(f"Video has been looped {loop_count} times.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output video path
    output_path = os.path.join(output_dir, f"{map_name}_stitched.mp4")

    # Write the final video to file
    try:
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f"Stitched video saved to '{output_path}'.")
    except Exception as e:
        print(f"Error: Failed to write stitched video to '{output_path}': {e}")
    finally:
        # Close all clips to release resources
        final_clip.close()
        for clip in clips:
            clip.close()
