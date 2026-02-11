import torch
import torch.nn.functional as F
from td3.td3_agent import TD3Agent

class TD3BCAgent(TD3Agent):
    def __init__(self, state_dim, action_dim, max_action,
                 device="cpu", lr=3e-4,
                 alpha=2.5, policy_delay=2):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=lr,
        )
        self.alpha = alpha
        self.policy_delay = policy_delay

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005, policy_refinement_factor=1):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.05).clamp(-0.2, 0.2)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1.0 - done) * gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        bc_loss = None
        if self.total_it % self.policy_delay == 0:
            pi = self.actor(state)
            q_pi, _ = self.critic(state, pi)

            alpha_refined = self.alpha * policy_refinement_factor
            lmbda = alpha_refined / q_pi.abs().mean().detach()

            td3_actor_loss = -lmbda * q_pi.mean()
            bc_loss = F.mse_loss(pi, action)
            actor_loss = td3_actor_loss + bc_loss

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return (
            critic_loss.item(),
            actor_loss.item() if actor_loss is not None else None,
            bc_loss.item() if bc_loss is not None else None,
        )
