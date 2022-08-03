import sys
import numpy as np
from DuDQNAgent import Agent
from utils import plot_learning_curve
from preprocess import make_env
import wandb
import torch.nn as nn

if __name__ == "__main__":
    env = make_env("PongNoFrameskip-v4")
    wandb.init(project="pong-dudqn")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 300
    agent = Agent(gamma=0.99, epsilon=0.1, lr= 0.0001, 
                input_dims=env.observation_space.shape, n_actions=env.action_space.n,
                memory_size=30000, eps_min=0.1, decay_rate=1e-5, batch_size=32,
                replace=1000, checkpoint_dir='models', algo="DuDQNAgent", env_name="PongNoFrameskip-v4")
    wandb.config = agent.__dict__
    agent.load_models()             # comment after training
    
    if load_checkpoint:
        agent.load_models()
    
    fname = f"{agent.algo}_{agent.env_name}_lr{agent.lr}_{n_games}games"
    figure_file = f"assets/{fname}.png"
    n_steps = 0
    scores, eps_history, steps_history = [], [], []
    
    # wandb.watch(agent.Q_eval, log="all")
    # wandb.watch(agent.Q_next, log="all")
    for i in range(1, n_games+1):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, int(done), observation_)
                agent.learn()
            observation = observation_
            n_steps += 1
            wandb.log({"score": score, "epsilon": agent.epsilon, "steps": n_steps, "games": i})

        scores.append(score)
        steps_history.append(n_steps)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if np.mean(scores[-100:]) >= np.mean(scores[-101:-1]):
            if not load_checkpoint:
                agent.save_models()
        if score > best_score:
            best_score = score
        

        sys.stdout.write(f"Game: {i}/{n_games} | Score: {score:.2f} | Average Score: {avg_score:.2f} | Best Score: {best_score} | Epsilon: {agent.epsilon:.2f} | steps: {n_steps}\n")
    plot_learning_curve(steps_history, scores, eps_history, figure_file)