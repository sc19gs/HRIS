import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
from scipy.stats import rice

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def secrecy_rate(ris_phases, ris_amp, bf_amp, bf_phase, n_RIS, n_b, n_e,
                 h0_b, h0_e, h_b, h_e, s_tx, H):
    """
    ris_phases: (N,)
    ris_amp:    (N,)
    bf_amp:     (M,1)  or (M,)
    bf_phase:   (M,1)  or (M,)
    h0_b,h0_e: (M,1) complex
    h_b,h_e:   (N,1) complex
    H:         (N,M) complex
    """
    # Ensure shapes
    bf_amp = np.asarray(bf_amp).reshape(-1, 1)
    bf_phase = np.asarray(bf_phase).reshape(-1, 1)

    theta = ris_amp * np.exp(1j * ris_phases)
    Theta = np.diag(theta)

    # Complex beamformer (normalize safely)
    w = (bf_amp + 1j * bf_phase) / np.sqrt(2.0)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-8:
        w = np.ones_like(w) / np.sqrt(len(w))
    else:
        w = w / norm_w

    # Signals
    s_b = (h0_b.conj().T @ w + h_b.conj().T @ Theta @ H @ w) * s_tx
    s_e = (h0_e.conj().T @ w + h_e.conj().T @ Theta @ H @ w) * s_tx

    n_b_eff = h_b.conj().T @ Theta @ n_RIS + n_b
    n_e_eff = h_e.conj().T @ Theta @ n_RIS + n_e

    signal_b = np.abs(s_b)**2
    signal_e = np.abs(s_e)**2
    noise_b  = np.abs(n_b_eff)**2
    noise_e  = np.abs(n_e_eff)**2

    eps = 1e-9
    gamma_b = signal_b / (noise_b + eps)
    gamma_e = signal_e / (noise_e + eps)

    Rs = np.log2(1.0 + gamma_b) - np.log2(1.0 + gamma_e)

    # sanitize to scalar float
    try:
        Rs = float(np.real(Rs))
    except Exception:
        Rs = -10.0
    if not np.isfinite(Rs):
        Rs = -10.0
    return Rs
    
def init_nRIS(N, sigma_r2=0.05):
    """Initialize the complex RIS noise for all N elements once."""
    n_RIS_full = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2)
    n_RIS_full = np.sqrt(sigma_r2) * n_RIS_full
    
    return n_RIS_full

def thermal_noise(sigma2=0.1):
    n_b = np.sqrt(sigma2/2) * (np.random.randn(1) + 1j*np.random.randn(1))
    n_e = np.sqrt(sigma2/2) * (np.random.randn(1) + 1j*np.random.randn(1))

    return n_b, n_e
    
def compute_hris_power(ris_amp, ris_modes, H, w, sigma_r2=0.05):
    """
    Compute HRIS power consumption based on RIS modes.
    ris_modes: 0=OFF, 1=PASSIVE, 2=ACTIVE
    """
    N, M = H.shape
    P_HRIS = 0.0
    w_norm_sq = np.linalg.norm(w)**2

    for n in range(N):
        h1n = H[n, :]
        norm_h1n_sq = np.linalg.norm(h1n)**2
        amp_sq = np.abs(ris_amp[n])**2

        if ris_modes[n] == 2:  # ACTIVE
            P_HRIS += amp_sq * (sigma_r2 + norm_h1n_sq * w_norm_sq)

        elif ris_modes[n] == 1:  # PASSIVE
            P_HRIS += amp_sq * (norm_h1n_sq * w_norm_sq)

    return P_HRIS

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.vstack, zip(*samples))
        # return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states)
        return (torch.tensor(states, dtype=torch.float32, device=device),
                torch.tensor(actions, dtype=torch.float32, device=device),
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.tensor(next_states, dtype=torch.float32, device=device))


    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, N, M, num_off=4):
        """
        Actor network with RIS phases, amplitudes (with Top-k OFF selection), 
        and normalized beamforming vector.

        Args:
            state_dim: dimension of state vector
            N: number of RIS elements
            M: number of beamforming antennas
            num_off: number of RIS elements to force into OFF mode (amp=0)
        """
        super(Actor, self).__init__()
        self.N = N
        self.M = M
        self.num_off = num_off

        hidden = 128
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # Heads
        self.ris_phase_head = nn.Linear(hidden, N)   # phases ∈ [0, 2π]
        self.ris_amp_head   = nn.Linear(hidden, N)   # amplitudes ∈ [0, 2], with top-k OFF
        self.bf_real_head   = nn.Linear(hidden, M)   # beamforming real part
        self.bf_imag_head   = nn.Linear(hidden, M)   # beamforming imag part

    def forward(self, state):
        # normalize input state
        norm = torch.norm(state, dim=-1, keepdim=True)
        state = state / (norm + 1e-8)

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # RIS phases in [0, 2π]
        ris_phases = 2 * torch.pi * torch.sigmoid(self.ris_phase_head(x))

        # RIS amplitudes in [0, 2]
        ris_amp_raw = self.ris_amp_head(x)
        ris_amplitudes = 2.0 * torch.sigmoid(ris_amp_raw)

        # === Top-k OFF selection ===
        # Select num_off lowest amplitudes → force to 0
        if self.num_off > 0:
            # Get indices of k smallest amplitudes
            _, off_indices = torch.topk(ris_amplitudes, self.num_off, dim=-1, largest=False)
            
            # Build mask (1 for active/passive, 0 for off)
            mask = torch.ones_like(ris_amplitudes)
            mask.scatter_(1, off_indices, 0.0)
            
            # Apply mask
            ris_amplitudes = ris_amplitudes * mask

        # Beamforming vector (complex, normalized)
        bf_real = self.bf_real_head(x)
        bf_imag = self.bf_imag_head(x)
        w = torch.complex(bf_real, bf_imag)  # (batch, M)

        # Normalize w (per batch element)
        w_norm = torch.linalg.norm(w, dim=-1, keepdim=True) + 1e-8
        w = w / w_norm

        return {
            "ris_phases": ris_phases,        # (batch, N)
            "ris_amplitudes": ris_amplitudes, # (batch, N) with num_off=0
            "beamforming": w                 # (batch, M) complex normalized
        }

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # normalize state
        norm = torch.norm(state, dim=-1, keepdim=True)
        state = state / (norm + 1e-8)
        
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Plot Functions
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def action_to_vector(action_dict):
    """
    Flattens structured action dict into a real-valued vector.
    RIS amplitudes + phases + beamforming real+imag.
    """
    ris_phases = action_dict["ris_phases"]        # (batch, N)
    ris_amp = action_dict["ris_amplitudes"]       # (batch, N)
    w = action_dict["beamforming"]                # (batch, M) complex

    bf_real = w.real
    bf_imag = w.imag

    return torch.cat([ris_phases, ris_amp, bf_real, bf_imag], dim=-1)

def classify_ris_elements(ris_amplitudes, tol=1e-4):
    """
    Classify RIS elements into OFF, PASSIVE, ACTIVE.
    ris_amplitudes: np.array of shape (N,)
    """
    off_idx = np.where(ris_amplitudes < 0.1)[0]
    passive_idx = np.where((ris_amplitudes >= 0.1) & (ris_amplitudes <= 1.0))[0]
    active_idx = np.where(ris_amplitudes > 1.0)[0]

    return off_idx, passive_idx, active_idx

def run_bandits(M, N, num_off, P_max_BS):
    
    # Hyperparameters
    # N = 10                  # RIS elements
    # M = 4                   # BS antennas
    b_param = 2.0  # non-centrality parameter
    scale_param = 1.0 # scale parameter
    state_dim = 2 * M + 2 * N + M * N 
    action_dim = 2 * N + 2 * M
    # num_off = 4             # fixed number of OFF elements
    num_episodes = 50000
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    noise_std = 0.50 
    noise_decay = 0.999
    min_noise_std = 0.15
    bandit_problems = 10
    # P_max_BS = 20
    
    # === Initialize networks ===
    actor = Actor(state_dim, N, M, num_off=num_off).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, N, M, num_off=num_off).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    
    # Tracking variables
    snr_history = []
    snr_histories = []
    critic_losses = []
    actor_losses = []
    
    # n_RIS = nRIS(N, 0.05)
    n_RIS_full = init_nRIS(N, sigma_r2=0.05)
    n_b, n_e = thermal_noise(0.1)
    
    for b in range(bandit_problems):
    
       # Fixed channels
        h0_b = (np.random.rayleigh(size = (M,1)) + 1j*np.random.rayleigh(size = (M,1))) / np.sqrt(2) # Base station to the legitimate user
        h0_e = (np.random.rayleigh(size = (M,1)) + 1j*np.random.rayleigh(size = (M,1))) / np.sqrt(2) # Base station to the eaverdropper
    
        h_b = (np.random.rayleigh(size = (N,1)) + 1j*np.random.rayleigh(size = (N,1))) / np.sqrt(2) # HRIS to the legitimate user 
        h_e = (np.random.rayleigh(size = (N,1)) + 1j*np.random.rayleigh(size = (N,1))) / np.sqrt(2) # HRIS to the eavesdropper  
    
    
        H = (rice.rvs(b_param, scale=scale_param, size=(N,M)) + 1j*rice.rvs(b_param, scale=scale_param, size=(N,M))) / np.sqrt(2) # Base station to the HRIS
    
    
        s_tx = (np.random.rayleigh(1) + 1j*np.random.rayleigh(1)) / np.sqrt(2)
    
        state_np = np.concatenate([h0_b.flatten(), h0_e.flatten(), H.flatten(), h_b.flatten(), h_e.flatten()])
        state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
    
        # Training Loop
        for episode in range(num_episodes):
    
            with torch.no_grad():
                action_dict = actor(state)
            action_vec = action_to_vector(action_dict)
            ris_amp_before_noise = action_vec.squeeze(0).detach().cpu().numpy()[N:2*N]
            # print("RIS amp before noise = ", ris_amp_before_noise)
            
            # add exploration noise (only on RIS params)
            noisy_vec = action_vec + noise_std * torch.randn_like(action_vec, device=device)
    
            # Convert for environment reward calc
            action_env = noisy_vec.squeeze(0).detach().cpu().numpy()
            ris_phases = action_env[:N] % (2*np.pi)
            ris_amp    = np.clip(action_env[N:2*N], 0, 2)
            
            bf_real = action_env[2*N:2*N+M]
            bf_imag = action_env[2*N+M:]
            w = bf_real + 1j*bf_imag

            # Enforce ∥w∥₂ ≤ √P_max_BS
            norm_w = np.linalg.norm(w)
            if norm_w**2 > P_max_BS:
                w = w / norm_w * np.sqrt(P_max_BS)
                    
            # Determine RIS modes
            ris_modes = np.zeros(N)
            ris_modes[ris_amp > 1.0] = 2      # Active
            ris_modes[(ris_amp >= 0.1) & (ris_amp <= 1.0)] = 1  # Passive

            # Compute HRIS power
            P_HRIS = compute_hris_power(ris_amp, ris_modes, H, w)

            # Enforce HRIS power constraint
            P_HRIS_max = 15000   # choose your HRIS power budget

            if P_HRIS > P_HRIS_max:
                scale = np.sqrt(P_HRIS_max / (P_HRIS + 1e-12))
                ris_amp = ris_amp * scale
            
            # n_RIS = nRIS(N, ris_amp, sigma_r2=0.05)
            active_mask = (ris_amp > 1.0).astype(float)
            n_RIS_step = n_RIS_full * active_mask.reshape(-1, 1)
    
            off_idx, passive_idx, active_idx = classify_ris_elements(ris_amp_before_noise)
            
            reward = secrecy_rate(ris_phases, ris_amp, w.real, w.imag,
                                  n_RIS_step, n_b, n_e, h0_b, h0_e, h_b, h_e, s_tx, H)
    
            
            # replay_buffer.push(state_np, action_env, [reward], state)
            replay_buffer.push(
                state.detach().cpu().numpy(),
                action_env,
                np.array([reward], dtype=np.float32),
                state.detach().cpu().numpy()
            )
    
            # Update networks if buffer is ready
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states = replay_buffer.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                with torch.no_grad():
                    # next_actions = target_actor(next_states)
                    next_actions_dict = target_actor(next_states)
                    next_actions = action_to_vector(next_actions_dict)
                    target_q = rewards + gamma * target_critic(next_states, next_actions)
    
                current_q = critic(states, actions)
                critic_loss = loss_fn(current_q, target_q)
                critic_losses.append(critic_loss.item())
    
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()
    
                # actor update
                actor_actions_dict = actor(states)
                actor_actions = action_to_vector(actor_actions_dict)
                actor_loss = -critic(states, actor_actions).mean()
                actor_losses.append(actor_loss.item())
    
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()
    
                # Soft target updates
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
            snr_history.append(reward)
            noise_std = max(noise_std * noise_decay, min_noise_std)
    
            if episode % 100 == 0:
                print(f"Episode {b}, Step {episode}, Reward: {reward:.3f}, Noise STD: {noise_std:.3f}, OFF elements = {off_idx}, PASSIVE elements = {passive_idx}, ACTIVE elements = {active_idx}")
                # print(f"Step {episode}, Reward: {reward:.3f}, Noise STD: {noise_std:.3f}")
    
        snr_histories.append(snr_history)
    
        # Plot 1: SNR with Moving Average
        # plt.figure(figsize=(12,6))
        # plt.plot(snr_history, label='Instantaneous Secrecy Rate', alpha=0.4)
        # plt.plot(moving_average(snr_history), label='Secrecy Rate Moving Avg (100 steps)', linewidth=2)
        # plt.xlabel('Steps')
        # plt.ylabel('Secrecy Rate')
        # plt.title('Secrecy Rate Convergence')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
    
        # # Plot 2: Critic Loss
        # plt.figure(figsize=(12,6))
        # plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
        # plt.xlabel('Training Steps')
        # plt.ylabel('Loss')
        # plt.title('Critic Loss Over Time')
        # plt.grid(True)
        # plt.show()
    
        # # Plot 3: Actor Loss
        # plt.figure(figsize=(12,6))
        # plt.plot(actor_losses, label='Actor Loss', alpha=0.7)
        # plt.xlabel('Training Steps')
        # plt.ylabel('Loss')
        # plt.title('Actor Loss Over Time')
        # plt.grid(True)
        # plt.show()
    
        snr_history = []
        # snr_histories = []
        critic_losses = []
        actor_losses = []

    snr_histories_mean = np.mean(snr_histories, axis=0)
    # Plot 1: SNR with Moving Average
    # plt.figure(figsize=(12,6))
    # plt.plot(snr_histories_mean, label='Instantaneous Secrecy Rate', alpha=0.4)
    # plt.plot(moving_average(snr_histories_mean), label='Secrecy Rate Moving Avg (100 steps)', linewidth=2)
    # plt.xlabel('Episode')
    # plt.ylabel('Secrecy rate')
    # plt.title('SNR Convergence with Moving Average')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return snr_histories_mean


if __name__ == "__main__":

    # M = [4, 5, 6, 7, 8, 9]
    # active_N = [4, 5, 6, 7, 8, 9]
    # Total_N = 14
    # P_max_BS = [20, 30, 40]

    # for i in M:
    #     for j in active_N:
    #         for k in P_max_BS:

    #             snr_histories_mean = run_bandits(i, Total_N, Total_N - j, k)

    #             with open("P"+str(k)+"_M"+str(i)+"_N"+str(j)+".txt", "w") as file:
    #                 file.write(','.join(str(item) for item in snr_histories_mean))

    # snr_histories_mean = run_bandits(4, 14, 14 - 9, 40)

    # with open("P"+str(40)+"_M"+str(4)+"_N"+str(9)+".txt", "w") as file:
    #     file.write(','.join(str(item) for item in snr_histories_mean))


    # M = [4, 5, 6, 7, 8, 9]
    # active_N = [6]
    # Total_N = 14
    # P_max_BS = [20, 30, 40]

    # for i in M:
    #     for j in active_N:
    #         for k in P_max_BS:

    #             snr_histories_mean = run_bandits(i, Total_N, Total_N - j, k)

    #             with open("P"+str(k)+"_M"+str(i)+"_N"+str(j)+".txt", "w") as file:
    #                 file.write(','.join(str(item) for item in snr_histories_mean))

    M = [8]
    active_N = [8]
    Total_N = 14
    P_max_BS = [20]

    for i in M:
        for j in active_N:
            for k in P_max_BS:

                snr_histories_mean = run_bandits(i, Total_N, Total_N - j, k)

                with open("P"+str(k)+"_M"+str(i)+"_N"+str(j)+".txt", "w") as file:
                    file.write(','.join(str(item) for item in snr_histories_mean))


    # snr_histories_mean = run_bandits(6, 14, 14 - 6, 20)

    # with open("P"+str(20)+"_M"+str(6)+"_N"+str(6)+".txt", "w") as file:
    #     file.write(','.join(str(item) for item in snr_histories_mean))

    # snr_histories_mean = run_bandits(7, 14, 14 - 6, 30)

    # with open("P"+str(30)+"_M"+str(7)+"_N"+str(6)+".txt", "w") as file:
    #     file.write(','.join(str(item) for item in snr_histories_mean))
    
    # snr_histories_mean = run_bandits(8, 14, 14 - 6, 40)

    # with open("P"+str(40)+"_M"+str(8)+"_N"+str(6)+".txt", "w") as file:
    #     file.write(','.join(str(item) for item in snr_histories_mean))
    
    # snr_histories_mean = run_bandits(4, 14, 14 - 6, 20)

    # with open("P"+str(20)+"_M"+str(4)+"_N"+str(6)+".txt", "w") as file:
    #     file.write(','.join(str(item) for item in snr_histories_mean))
