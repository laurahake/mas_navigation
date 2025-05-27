import pickle
from custom_environment import raw_env
import numpy as np
from simple_pid import PID
import hashlib
import pygame

def stable_hash(state, num_states=5_000_000):
    state = np.array(state, dtype=np.int32)
    state_bytes = state.tobytes()
    return int(hashlib.sha256(state_bytes).hexdigest(), 16) % num_states

def print_state(state, size=9):
    print("=== Q-State Representation ===")
    
    # Extract segments
    center = state[:9]  # 3x3 fine grid
    top = state[9]
    left = state[10]
    right = state[11]
    bottom = state[12]
    last_action = state[13] if len(state) > 13 else None
    
    # Print layout:
    #     [ T ]
    # [L] [Center 3x3] [R]
    #     [ B ]
    
    print(f"   Top: {top}")
    print()
    
    print("Center Grid:")
    for r in range(2, -1, -1):
        row = center[r*3:(r+1)*3]
        print("        " + " ".join(f"{v:2}" for v in row))
        
    print()
    print(f"Left: {left}       Right: {right}")
    print()
    print(f"  Bottom: {bottom}")
    
    if last_action is not None:
        action_map = {0: "↑ up", 1: "↓ down", 2: "← left", 3: "→ right"}
        print()
        print(f"Last Action: {last_action} ({action_map.get(last_action, 'unknown')})")

    print("==============================\n")


with open("training_data.pkl", "rb") as f:
    Q, TD_error_per_episode, reward_per_episode = pickle.load(f)

def evaluate_policy(Q, episodes=1, seed=42):
    env = raw_env(render_mode="human")
    total_rewards = []
    num_states = Q.shape[0]

    agent_states, agent_observations = env.reset(seed=seed)
    agent_done = {agent: False for agent in env.agents}
    total_reward = 0

    while not all(agent_done[agent] for agent in agent_done):
        env.agent_selection = env.agents[0]

        for agent in env.agents:
            if env.terminations[agent]:
                action = None
            else:
                state = agent_states[agent]
                observation = agent_observations[agent]
                idx = stable_hash(state) % num_states
                q_vals = Q[idx]
                max_q = np.max(q_vals)
                best_actions = np.flatnonzero(q_vals == max_q)
                discrete_action = np.random.choice(best_actions)
                action = env.get_cont_action(observation, env.world.dim_p, discrete_action, agent)
                print(f"{agent}: is taking  {discrete_action}")
            env.step(action)
        pygame.event.pump()

        env.agent_selection = env.agents[0]
        for agent in env.agents:
            if agent_done[agent]:
                env.next_agent()
                continue
            observation, reward, termination, truncation, info, state = env.last()
            agent_observations[agent] = observation
            agent_states[agent] = state
            if termination or truncation:
                agent_done[agent] = True
            total_reward += reward
            env.next_agent()

        total_rewards.append(total_reward)
        print(f"Episode abgeschlossen mit Reward: {total_reward:.2f}")

    env.close()
    return total_rewards

evaluate_policy(Q, episodes=5)