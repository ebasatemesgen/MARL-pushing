#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
import sys
current_dir = os.getcwd()
import imageio
# You can add any directory to the path, here's how you add the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pathlib import Path
from typing import Union, Set

import cv2

from utils import EvaluationUtils, PathUtils, InjectMode


def render(
    checkpoint_path: Union[str, Path],
    n_episodes: int,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):
    print(checkpoint_path)
    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint( checkpoint_path)

    inject = agents_to_inject is not None and len(agents_to_inject) > 0
    rewards, best_gif, _, _ = EvaluationUtils.rollout_episodes(
        n_episodes=n_episodes,
        render=True,
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

    save_dir = PathUtils.result_dir / f"{env_title}/{model_title}/gifs"
    save_dir.mkdir(parents=True, exist_ok=True)  # This will create the directory if it doesn't exist
    
    name = f"{model_name}_{env_name}" + ("_" + inject_name if inject else "")

    gif_path = save_dir / f"{name}.gif"
    resized_gif = [cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)) for img in best_gif]
    
    # Save the images as a GIF
    imageio.mimsave(gif_path, resized_gif, duration=1/20, loop=0)  # Assuming 30 FPS to loop add this loop=0

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("No checkpoint path passed as arg.")
    #     sys.exit(-1)

    # checkpoint_path = "/home/sarah/Desktop/Eba_ws/HetGPPO/scratch/ray_results/transport/HetGPPO/MultiPPOTrainer_transport_123d2_00000_0_2024-11-24_21-33-43/checkpoint_000010"
    
    checkpoint_path = "/home/sarah/Desktop/Eba_ws/HetGPPO/scratch/ray_results/transport/HetGPPO/MultiPPOTrainer_transport_893e2_00000_0_2024-11-24_21-58-31/checkpoint_000027"
    print(checkpoint_path)

    render(
        checkpoint_path=checkpoint_path,
        n_episodes=5,
        agents_to_inject=None,
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.5,
    )