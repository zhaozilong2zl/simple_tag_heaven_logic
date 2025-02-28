import torch
import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, VmasEnv
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator import rendering

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_good_agents = kwargs.pop("num_good_agents", 1)
        num_adversaries = kwargs.pop("num_adversaries", 3)
        num_landmarks = kwargs.pop("num_landmarks", 2)
        self.shape_agent_rew = kwargs.pop("shape_agent_rew", False)
        self.shape_adversary_rew = kwargs.pop("shape_adversary_rew", False)
        self.agents_share_rew = kwargs.pop("agents_share_rew", False)
        self.adversaries_share_rew = kwargs.pop("adversaries_share_rew", True)
        self.observe_same_team = kwargs.pop("observe_same_team", True)
        self.observe_pos = kwargs.pop("observe_pos", True)
        self.observe_vel = kwargs.pop("observe_vel", True)
        self.bound = kwargs.pop("bound", 1.0)
        self.respawn_at_catch = kwargs.pop("respawn_at_catch", False)
        # add Heaven reward/penalty
        self.adv_heaven_reward = kwargs.pop("adv_heaven_reward", 10.0)  # one time reward for catching
        self.agent_heaven_penalty = kwargs.pop("agent_heaven_penalty", -10.0)  # one time penalty for being caught
        self.heaven_size = 0.1  # heaven is drawn as a circle
        self.heaven_position = torch.tensor([0.0, self.bound + self.heaven_size*2], device=device, dtype=torch.float32)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.visualize_semidims = False

        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=self.bound,
            y_semidim=self.bound,
            substeps=10,
            collision_force=500,
        )
        # set any world properties first
        num_agents = num_adversaries + num_good_agents
        self.adversary_radius = 0.075

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            name = f"adversary_{i}" if adversary else f"agent_{i - num_adversaries}"
            agent = Agent(
                name=name,
                render_action=True,
                collide=True,
                shape=Sphere(radius=self.adversary_radius if adversary else 0.05),
                u_multiplier=1.0 if adversary else 1.0,
                max_speed=0.5 if adversary else 0.5,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
            )

            # agent active if not collide with other team
            agent.active = torch.ones(batch_dim, device=device, dtype=torch.bool)
            # attribute to record whether heaven reward has been given in a single episode
            agent.heaven_reward_given = torch.zeros(batch_dim, device=device, dtype=torch.bool)

            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:

            # reset active flag for vectorized environments, according to specific env_index
            if env_index is not None:
                agent.active[env_index] = True
                agent.heaven_reward_given[env_index] = False
            else:
                agent.active[:] = True
                agent.heaven_reward_given[:] = False

            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.bound,
                    self.bound,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -(self.bound - 0.1),
                    self.bound - 0.1,
                ),
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        # ignore collisions between inactive agents
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        dist_min = agent1.shape.radius + agent2.shape.radius
        return (dist < dist_min) & agent1.active & agent2.active

    # detect all collisions in the world
    def detect_collisions(self):
        adversaries = self.adversaries()
        good_agents = self.good_agents()

        # abstract all agents' positions/active status
        adv_pos = torch.stack([a.state.pos for a in adversaries], dim=1)  # (batch_dim, n_adv, 2)
        good_pos = torch.stack([a.state.pos for a in good_agents], dim=1)  # (batch_dim, n_good, 2)
        adv_active = torch.stack([a.active for a in adversaries], dim=1)  # (batch_dim, n_adv)
        good_active = torch.stack([a.active for a in good_agents], dim=1)  # (batch_dim, n_good)

        # calculate the distance between each pair of agents
        delta_pos = adv_pos.unsqueeze(2) - good_pos.unsqueeze(1)  # (batch_dim, n_adv, n_good, 2)
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)  # (batch_dim, n_adv, n_good)
        dist_min = adversaries[0].shape.radius + good_agents[0].shape.radius  # homo agents

        # consider active
        active_mask = adv_active.unsqueeze(2) & good_active.unsqueeze(1)  # (batch_dim, n_adv, n_good)
        collisions = (dist < dist_min) & active_mask  # (batch_dim, n_adv, n_good)

        return collisions  # collision matrix

    # process collision
    def process_collisions(self, collisions):
        adversaries = self.adversaries()
        good_agents = self.good_agents()
        if not collisions.any():
            return False

        # detect if every agent at least collide with one agent belongs to the opposite team
        adv_collided = collisions.any(dim=2)  # (batch_dim, n_adv)
        good_collided = collisions.any(dim=1)  # (batch_dim, n_good)

        # apply one time reward/penalty for agent who haven't get heaven reward this episode
        for i, adv in enumerate(adversaries):
            # just apply reward to those who just collide and haven't get heaven reward yet
            new_collisions = adv_collided[:, i] & (~adv.heaven_reward_given)
            if new_collisions.any():
                if not hasattr(adv, 'rew'):
                    adv.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                # apply one time heaven reward for adversaries
                adv.rew = torch.zeros_like(adv.rew)
                adv.rew[new_collisions] += self.adv_heaven_reward
                # mark heaven reward has been given
                adv.heaven_reward_given[new_collisions] = True

        for j, agent in enumerate(good_agents):
            new_collisions = good_collided[:, j] & (~agent.heaven_reward_given)
            if new_collisions.any():
                if not hasattr(agent, 'rew'):
                    agent.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)
                # apply one time heaven penalty for agents
                agent.rew = torch.zeros_like(agent.rew)
                agent.rew[new_collisions] += self.agent_heaven_penalty
                # mark heaven penalty has been given
                agent.heaven_reward_given[new_collisions] = True

        # update active status to show if an agent is in heaven
        # if an agent is in heaven, set its position to heaven position, and velocity to 0
        for i, adv in enumerate(adversaries):
            mask = adv_collided[:, i]
            if mask.any():
                adv.active[mask] = False
                adv.state.pos[mask] = self.heaven_position  # heaven position
                adv.state.vel[mask] = 0.0
        for j, agent in enumerate(good_agents):
            mask = good_collided[:, j]
            if mask.any():
                agent.active[mask] = False
                agent.state.pos[mask] = self.heaven_position  # heaven position
                agent.state.vel[mask] = 0.0

        return True

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:

            collisions = self.detect_collisions()  # detect all collisions
            self.process_collisions(collisions)  # process all possible collision

            for a in self.world.agents:
                if a.active.any():  # if any env in batch_dim has active agent
                    active_mask = a.active
                    if a.adversary:
                        normal_rew = self.adversary_reward(a)
                    else:
                        normal_rew = self.agent_reward(a)

                    if not hasattr(a, 'rew'):
                        a.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

                    # count active agents' normal rew(except heaven reward) only
                    if active_mask.any():
                        a.rew[active_mask] = normal_rew[active_mask]
            self.agents_rew = torch.stack([a.rew for a in self.good_agents()], dim=-1).sum(-1)
            self.adverary_rew = torch.stack([a.rew for a in self.adversaries()], dim=-1).sum(-1)
            if self.respawn_at_catch:
                for a in self.good_agents():
                    for adv in self.adversaries():
                        coll = self.is_collision(a, adv)
                        a.state.pos[coll] = torch.zeros(
                            (self.world.batch_dim, self.world.dim_p),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-self.bound, self.bound,)[coll]
                        a.state.vel[coll] = 0.0

        if agent.adversary:
            if self.adversaries_share_rew:
                return self.adverary_rew
            else:
                return agent.rew
        else:
            if self.agents_share_rew:
                return self.agents_rew
            else:
                return agent.rew

    def agent_reward(self, agent: Agent):
        # Agents are negatively rewarded if caught by adversaries
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        adversaries = self.adversaries()
        if self.shape_agent_rew:
            # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * torch.linalg.vector_norm(
                    agent.state.pos - adv.state.pos, dim=-1
                )
        if agent.collide:
            for a in adversaries:
                rew[self.is_collision(a, agent)] -= 10

        return rew

    def adversary_reward(self, agent: Agent):
        # Adversaries are rewarded for collisions with agents
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        agents = self.good_agents()
        if (
            self.shape_adversary_rew
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                a.state.pos - agent.state.pos,
                                dim=-1,
                            )
                            for a in agents
                        ],
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
            )
        if agent.collide:
            # for ag in agents:
            #     rew[self.is_collision(ag, agent)] += 10
            pass
        return rew

    def observation(self, agent: Agent):
        # if inactive (in heaven), return all zeros observation
        obs_dim = self._get_obs_dim(agent)
        final_obs = torch.zeros(
            (self.world.batch_dim, obs_dim),
            device=self.world.device,
            dtype=torch.float32,
        )
        if (~agent.active).any():
            agent.state.pos[~agent.active] = self.heaven_position
            agent.state.vel[~agent.active] = 0.0

        if (~agent.active).all():
            return final_obs

        if agent.active.any():

            # get positions of all entities in this agent's reference frame
            entity_pos = []
            for entity in self.world.landmarks:
                entity_pos.append(entity.state.pos - agent.state.pos)

            other_pos = []
            other_vel = []
            for other in self.world.agents:
                if other is agent:
                    continue

                # only add actual pos and vel to other agents when they active
                # otherwise, set them to 0
                zero_pos = torch.zeros_like(other.state.pos)
                zero_vel = torch.zeros_like(other.state.vel)

                if agent.adversary and not other.adversary:
                    # Check whether "other" is active for each environment instance separately
                    # Create a active mask tensor with a shape consistent with batch_dim
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    # only add actual pos and vel to other agents when they active
                    # otherwise, set them to 0
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    vel = torch.where(active_mask, other.state.vel, zero_vel)
                    other_pos.append(pos)
                    other_vel.append(vel)
                elif not agent.adversary and not other.adversary and self.observe_same_team:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    vel = torch.where(active_mask, other.state.vel, zero_vel)
                    other_pos.append(pos)
                    other_vel.append(vel)
                elif not agent.adversary and other.adversary:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    other_pos.append(pos)
                elif agent.adversary and other.adversary and self.observe_same_team:
                    active_mask = other.active.unsqueeze(-1).expand_as(other.state.pos)
                    pos = torch.where(active_mask, other.state.pos - agent.state.pos, zero_pos)
                    other_pos.append(pos)

            normal_obs = torch.cat(
                [
                    *([agent.state.vel] if self.observe_vel else []),
                    *([agent.state.pos] if self.observe_pos else []),
                    *entity_pos,
                    *other_pos,
                    *other_vel,
                ],
                dim=-1,
            )
            final_obs[agent.active] = normal_obs[agent.active]

        return final_obs

    def _get_obs_dim(self, agent):
        """get the dimension of observation"""
        # get obs dimension
        dim = 0
        # self vel and pos
        if self.observe_vel:
            dim += self.world.dim_p
        if self.observe_pos:
            dim += self.world.dim_p
        # Landmarks
        dim += len(self.world.landmarks) * self.world.dim_p

        # other agents
        for other in self.world.agents:
            if other is agent:
                continue
            if agent.adversary and not other.adversary:
                dim += self.world.dim_p
                dim += self.world.dim_p
            elif not agent.adversary and not other.adversary and self.observe_same_team:
                dim += self.world.dim_p
                dim += self.world.dim_p
            elif not agent.adversary and other.adversary:
                dim += self.world.dim_p
            elif agent.adversary and other.adversary and self.observe_same_team:
                dim += self.world.dim_p

        return dim
    
    def done(self):
        # Get the active status of all good agents and adversaries
        good_active = torch.stack([agent.active for agent in self.good_agents()], dim=0)  # (n_good, batch_dim)
        adv_active = torch.stack([agent.active for agent in self.adversaries()], dim=0)  # (n_adv, batch_dim)

        # In each environment, all good agents or all adversaries are inactive
        all_good_inactive = ~good_active.any(dim=0)  # (batch_dim,)
        all_adv_inactive = ~adv_active.any(dim=0)  # (batch_dim,)

        # done status in each vectorize environment
        return all_good_inactive | all_adv_inactive  # (batch_dim,)

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Draw Heaven at self.heaven_position
        heaven_geom = rendering.make_circle(self.heaven_size)
        xform = rendering.Transform()
        heaven_geom.add_attr(xform)
        xform.set_translation(self.heaven_position[0].item(), self.heaven_position[1].item())
        heaven_geom.set_color(0.9, 0.9, 0.2)  # Golden Heaven
        geoms.append(heaven_geom)

        # Perimeter
        for i in range(4):
            geom = Line(
                length=2
                * ((self.bound - self.adversary_radius) + self.adversary_radius * 2)
            ).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 0
                        else -self.bound - self.adversary_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.bound + self.adversary_radius
                        if i == 1
                        else -self.bound - self.adversary_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)
        return geoms

    

if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
