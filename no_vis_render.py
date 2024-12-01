import os
import sys
import imageio
from pathlib import Path
from typing import Union, Set
import cv2

from utils import EvaluationUtils, PathUtils, InjectMode


def render_and_save_gif(
    checkpoint_path: Union[str, Path],
    n_episodes: int,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):
    """
    Render episodes from a checkpoint and save them as a GIF.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    inject = agents_to_inject is not None and len(agents_to_inject) > 0

    # Perform rollouts with rendering enabled
    rewards, best_gif, _, _ = EvaluationUtils.rollout_episodes(
        n_episodes=n_episodes,
        render=True,  # Enable rendering to capture frames
        get_obs=False,
        get_actions=False,
        trainer=trainer,
        env=env,
        inject=inject,
        inject_mode=inject_mode,
        noise_delta=noise_delta,
        agents_to_inject=agents_to_inject,
        use_pickle=False,
    )

    (
        model_title,
        model_name,
        env_title,
        env_name,
    ) = EvaluationUtils.get_model_name(config)

    inject_title, inject_name = EvaluationUtils.get_inject_name(
        agents_to_inject=agents_to_inject,
        noise_delta=noise_delta,
        inject_mode=inject_mode,
    )

    # Define the directory to save GIFs
    save_dir = PathUtils.result_dir / f"{env_title}/{model_title}/gifs"
    save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # File name for the GIF
    name = f"{model_name}_{env_name}" + ("_" + inject_name if inject else "")
    gif_path = save_dir / f"{name}.gif"

    # Resize frames and save as a GIF
    resized_gif = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) for img in best_gif]
    imageio.mimsave(gif_path, resized_gif, duration=1 / 20, loop=0)  # Assuming 20 FPS

    print(f"GIF saved at: {gif_path}")


if __name__ == "__main__":
    checkpoint_path = "/home/sarah/Desktop/Eba_ws/HetGPPO/scratch/ray_results/transport/HetGPPO/MultiPPOTrainer_transport_893e2_00000_0_2024-11-24_21-58-31/checkpoint_000027"
    print(f"Checkpoint path: {checkpoint_path}")

    render_and_save_gif(
        checkpoint_path=checkpoint_path,
        n_episodes=5,
        agents_to_inject=None,
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.5,
    )
