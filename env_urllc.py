import gymnasium as gym
import numpy as np
from scipy.special import erf

class ENV_paper(gym.Env):
    def __init__(self, lambda_rate, D_max, xi, max_power, snr_feedback, harq_type):
        # Tham số hệ thống
        self.lambda_rate = lambda_rate  # Average arrival rate (bits/slot)
        self.D_max = D_max  # Max delay (slots)
        self.xi = xi  # Target delay violation probability
        self.max_power = max_power  # Max transmit power
        self.snr_feedback = snr_feedback  # Use SNR feedback or not
        self.harq_type = harq_type  # 'CC' or 'IR'
        self.n = 200  # Channel uses per slot
        self.T = int(10 / xi)  # Time slots per episode
        self.Delta = 20 * max_power  # Penalty coefficient
        self.beta = 16  # Penalty exponent

        # State space: [queue_length, arrival_rate, delay_violations, transmission_count, (residual_SNR)]
        state_dim = 4 if not snr_feedback else 5
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )

        # Action space: [coding_rate, transmit_power]
        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 0.001]), 
            high=np.array([np.inf, max_power]), 
            dtype=np.float32
        )

        # Initialize environment
        self.reset()

    def reset(self, seed=None):
        # Initialize random generator
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state variables
        self.q_t = 0  # Queue length
        self.A_t = np.random.poisson(self.lambda_rate)  # New arrivals
        self.d_t = 0  # Delay violation count
        self.k = 0  # Transmission count (0 = new packet)
        self.t = 0  # Current timestep
        self.gamma_k = 0.0  # Residual SNR
        self.arrival_history = [self.A_t]  # Arrival history for PODD
        self.previous_snrs = []  # SNR history for HARQ
        
        # HARQ-specific states
        self.initial_R = None  # Initial coding rate for the packet
        self.initial_power = None  # Initial power for CC without feedback
        
        # Build initial state
        state = [self.q_t, self.A_t, self.d_t, self.k]
        if self.snr_feedback:
            state.append(self.gamma_k)
            
        return np.array(state), {}

    def step(self, action):
        R_t, p_t = action
        self.t += 1

        # Clip actions to valid range
        R_t = max(R_t, 0.1)
        p_t = np.clip(p_t, 0.001, self.max_power)
        
        # KÊNH TRUYỀN
        """
        Tổn hao truyền dẫn: h_l
        Trải rộng chùm tia: h_bl
        Nhiễu loạn: h_t
        => Hệ số kênh h = h_l * h_bl * h_t
        """
        # Tham số
        distance = 35 # khoảng cách từ máy phát tới máy thu (m)
        extinction_coe = {'pure sea': 0.05, 'clear ocean': 0.15, 'coastal water': 0.31, 'habor': 2.17} # (m^-1)
        a  = 0.1 # bán kính máy thu (m)
        W_L = 4 # beam width (m)
        r = 0 # khoảng cách tù tâm chùm beam tới thấu kính thu (m)
        sigma_T2 = (10**-3)**2 # Nhiễu loạn mạnh
        mu_T = sigma_T2 / (-2)
        gain = 10 # dB

        # Tổn hao truyền dẫn
        h_l = np.exp(-distance * extinction_coe['pure sea'])
        # Trải rộng chùm tia
        v = np.sqrt(np.pi) * a / (np.sqrt(2) * W_L)
        A_0 = erf(v)**2
        W_Leq = W_L**2 * (np.sqrt(np.pi) * erf(v)) / (2 * v * np.exp(-v**2))
        h_bl = A_0 * np.exp(-2 * r**2 / W_Leq)
        # Nhiễu loạn
        h_t = np.random.lognormal(mean=mu_T, sigma=np.sqrt(sigma_T2))
        # h_t = np.exp(X)
        # Hệ số kênh
        h = 10**(gain/10) * h_l * h_t * h_bl
        snr = p_t * (np.abs(h)**2) /(10**(-14))

        # Determine target coding rate
        if self.k == 0:  # New transmission
            self.initial_R = R_t  # Save initial coding rate
            self.initial_power = p_t  # Save initial power
            target_R = R_t
        else:  # Retransmission
            target_R = self.initial_R  # Use initial coding rate

        # Check transmission success based on HARQ mechanism
        success = False
        
        if self.k == 0:  # New transmission
            success = np.log2(1 + snr) >= target_R
        else:  # Retransmission
            if self.harq_type == 'CC':
                if self.snr_feedback:  # CC-HARQ with feedback
                    accumulated_snr = sum(self.previous_snrs) + snr
                    success = np.log2(1 + accumulated_snr) >= target_R
                else:  # CC-HARQ without feedback
                    # Use initial power for retransmission
                    snr_fixed = self.initial_power * (np.abs(h)**2)
                    accumulated_snr = sum(self.previous_snrs) + snr_fixed
                    success = np.log2(1 + accumulated_snr) >= target_R
                    
            elif self.harq_type == 'IR' and self.snr_feedback:  # IR-HARQ
                total_rate = sum([np.log2(1 + s) for s in self.previous_snrs]) + np.log2(1 + snr)
                success = total_rate >= target_R

        # Calculate service rate (bits served)
        S_t = self.n * target_R if success else 0

        # Update temporary queue length
        q_tmp = max(self.q_t + self.A_t - S_t, 0)
        
        # Tính q_th(t) dựa trên lịch sử A(t) trong D_max slot
        if len(self.arrival_history) >= self.D_max:
            q_th = sum(self.arrival_history[-self.D_max:])
        else:
            q_th = sum(self.arrival_history)

        # Check delay violation
        delay_violation = q_tmp > q_th
        if delay_violation:
            self.d_t += 1
            w_t = self._calculate_penalty()
            reward = -p_t - w_t
        else:
            reward = -p_t

        # Apply PODD (Proactive Outdated Data Dropping)
        self.q_t = min(q_tmp, q_th)

        # Generate new arrivals for next slot
        self.A_t = np.random.poisson(self.lambda_rate)
        self.arrival_history.append(self.A_t)
        if len(self.arrival_history) > self.D_max:  # Keep reasonable history
            self.arrival_history.pop(0)

        # Update transmission state
        if success:
            # Reset for new packet
            self.k = 0
            self.gamma_k = 0.0
            self.previous_snrs = []
            self.initial_R = None
            self.initial_power = None
        else:
            self.k += 1
            self.previous_snrs.append(snr)
            
            # Update residual SNR if feedback is available
            if self.snr_feedback:
                if self.harq_type == 'CC':
                    # Sum of previous SNRs (excluding current)
                    prev_sum = sum(self.previous_snrs[:-1]) if self.k > 1 else 0
                    self.gamma_k = max(2**target_R - 1 - prev_sum, 0)
                elif self.harq_type == 'IR':
                    # Product of (1 + SNR) for previous transmissions
                    prev_snrs = self.previous_snrs[:-1] if self.k > 1 else []
                    product = np.prod([1 + s for s in prev_snrs]) if prev_snrs else 1
                    self.gamma_k = max(2**target_R / product - 1, 0)

        # Build next state
        state = [self.q_t, self.A_t, self.d_t, self.k]
        if self.snr_feedback:
            state.append(self.gamma_k)

        # Check episode termination
        terminated = self.t >= self.T
        truncated = False  # We don't use truncation in this environment
        
        # Additional info for debugging
        info = {
            'power': p_t,
            'rate': target_R,
            'success': success,
            'delay_violation': delay_violation,
            'total_delay_violations': self.d_t,
            'transmission_count': self.k
        }

        # Return 4 values for backward compatibility
        done = terminated or truncated
        return np.array(state), reward, done, info

    def _calculate_penalty(self):
        """Calculate penalty based on delay violations"""
        target_violations = self.T * self.xi
        
        if self.d_t <= target_violations:
            return self.Delta * (self.d_t / target_violations) ** self.beta
        return self.Delta