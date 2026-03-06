import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# -----------------------------
# Environment / scripted B types
# -----------------------------
ACTIONS = ["probe", "trust", "wait"]
B_ACTIONS = ["reply_fast", "reply_slow", "no_reply"]


@dataclass
class BenignB:
    """Consistent type."""
    def act(self, t, a_action):
        if a_action == "probe":
            return "reply_fast"
        if a_action == "trust":
            return "reply_fast"
        return "reply_slow"


@dataclass
class AdversarialB:
    """Looks benign early, then violates expectations."""
    switch_time: int = 4

    def act(self, t, a_action):
        if t < self.switch_time:
            # benign-looking prefix
            if a_action == "probe":
                return "reply_fast"
            if a_action == "trust":
                return "reply_fast"
            return "reply_slow"
        # later deception / mismatch
        if a_action == "probe":
            return "no_reply"
        if a_action == "trust":
            return "reply_slow"
        return "no_reply"


# ---------------------------------
# A's shallow model of B (DoM = 1)
# ---------------------------------
def policy_theta_benign(a_action):
    # P(B_action | benign, A_action)
    if a_action in ["probe", "trust"]:
        return {
            "reply_fast": 0.85,
            "reply_slow": 0.10,
            "no_reply": 0.05,
        }
    return {
        "reply_fast": 0.10,
        "reply_slow": 0.80,
        "no_reply": 0.10,
    }


def policy_theta_adversarial_model(a_action):
    """
    A's *modeled* adversarial type.
    Still only shallow: assumes a fixed policy, not a deceptive phase shift.
    """
    if a_action == "probe":
        return {
            "reply_fast": 0.55,
            "reply_slow": 0.30,
            "no_reply": 0.15,
        }
    if a_action == "trust":
        return {
            "reply_fast": 0.60,
            "reply_slow": 0.25,
            "no_reply": 0.15,
        }
    return {
        "reply_fast": 0.20,
        "reply_slow": 0.60,
        "no_reply": 0.20,
    }


TYPE_MODELS = {
    "benign": policy_theta_benign,
    "adversarial": policy_theta_adversarial_model,
}


# -----------------------------
# Reward model
# -----------------------------
def reward_fn(a_action, b_action):
    """
    Abstract reward:
    +1 if expectation broadly met
    -1 if violated

    Here: reply_fast is 'good', reply_slow or no_reply are progressively worse.
    """
    if a_action == "trust":
        return 1 if b_action == "reply_fast" else -1
    if a_action == "probe":
        return 1 if b_action in ["reply_fast", "reply_slow"] else -1
    if a_action == "wait":
        return 1 if b_action == "reply_slow" else -1
    return -1


# -----------------------------
# Agent A with ℵ-style detector
# -----------------------------
@dataclass
class AlephAgent:
    belief: dict = field(default_factory=lambda: {"benign": 0.5, "adversarial": 0.5})
    delta: float = 0.12     # behavioral mismatch threshold
    omega: float = 2.0      # reward mismatch threshold on cumulative reward
    ruled_out: dict = field(default_factory=lambda: {"benign": False, "adversarial": False})
    cumulative_reward: float = 0.0
    expected_cumulative_reward: float = 0.0
    oob: bool = False

    belief_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    expected_reward_history: list = field(default_factory=list)
    anomaly_history: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    b_action_history: list = field(default_factory=list)

    def active_types(self):
        return [k for k, v in self.ruled_out.items() if not v]

    def normalize_belief(self):
        total = sum(self.belief[k] for k in self.active_types())
        if total == 0:
            return
        for k in self.belief:
            self.belief[k] = self.belief[k] / total if not self.ruled_out[k] else 0.0

    def choose_action(self, t):
        if self.oob:
            # OOB policy: defensive / disengage
            return "wait"

        # Simple planner:
        # trust when benign belief is high, otherwise probe
        if self.belief["benign"] >= 0.7:
            return "trust"
        return "probe"

    def expected_action_distribution(self, a_action):
        dist = {b_act: 0.0 for b_act in B_ACTIONS}
        for theta in self.active_types():
            p_theta = self.belief[theta]
            model_dist = TYPE_MODELS[theta](a_action)
            for b_act in B_ACTIONS:
                dist[b_act] += p_theta * model_dist[b_act]
        return dist

    def expected_reward(self, a_action):
        dist = self.expected_action_distribution(a_action)
        return sum(reward_fn(a_action, b_act) * p for b_act, p in dist.items())

    def update(self, t, a_action, observed_b_action):
        anomaly = {
            "behavior": False,
            "reward": False,
            "ruled_out_now": [],
            "oob_switch": False,
        }

        # 1) expected likelihood under current belief mixture
        mixture = self.expected_action_distribution(a_action)
        L = mixture[observed_b_action]

        # 2) rule out types whose individual likelihood is too small
        for theta in self.active_types():
            l_theta = TYPE_MODELS[theta](a_action)[observed_b_action]
            if l_theta < self.delta:
                self.ruled_out[theta] = True
                anomaly["behavior"] = True
                anomaly["ruled_out_now"].append(theta)

        # 3) Bayesian update over remaining types
        for theta in self.active_types():
            self.belief[theta] *= TYPE_MODELS[theta](a_action)[observed_b_action]
        self.normalize_belief()

        # 4) reward mismatch
        r = reward_fn(a_action, observed_b_action)
        er = self.expected_reward(a_action)

        self.cumulative_reward += r
        self.expected_cumulative_reward += er

        if abs(self.cumulative_reward - self.expected_cumulative_reward) > self.omega:
            anomaly["reward"] = True

        # 5) stopping rule: all types ruled out => OOB
        if len(self.active_types()) == 0:
            self.oob = True
            anomaly["oob_switch"] = True

        # logging
        self.belief_history.append(self.belief.copy())
        self.reward_history.append(self.cumulative_reward)
        self.expected_reward_history.append(self.expected_cumulative_reward)
        self.anomaly_history.append(anomaly)
        self.action_history.append(a_action)
        self.b_action_history.append(observed_b_action)

        return L, r, er, anomaly


# -----------------------------
# Simulation
# -----------------------------
def run_episode(true_B, T=10, delta=0.12, omega=2.0):
    agent = AlephAgent(delta=delta, omega=omega)

    for t in range(T):
        a_action = agent.choose_action(t)
        b_action = true_B.act(t, a_action)
        agent.update(t, a_action, b_action)

    return agent


# -----------------------------
# Plotting
# -----------------------------
def plot_run(agent, title="Run"):
    ts = np.arange(len(agent.belief_history))
    benign_belief = [b["benign"] for b in agent.belief_history]
    adv_belief = [b["adversarial"] for b in agent.belief_history]

    # Plot 1: belief over time
    plt.figure(figsize=(7, 4))
    plt.plot(ts, benign_belief, label="P(benign)")
    plt.plot(ts, adv_belief, label="P(adversarial)")
    for t, an in enumerate(agent.anomaly_history):
        if an["behavior"] or an["reward"] or an["oob_switch"]:
            plt.axvline(t, linestyle="--", alpha=0.4)
    plt.xlabel("t")
    plt.ylabel("belief")
    plt.title(f"{title}: beliefs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: cumulative reward
    plt.figure(figsize=(7, 4))
    plt.plot(ts, agent.reward_history, label="observed cumulative reward")
    plt.plot(ts, agent.expected_reward_history, label="expected cumulative reward")
    for t, an in enumerate(agent.anomaly_history):
        if an["behavior"] or an["reward"] or an["oob_switch"]:
            plt.axvline(t, linestyle="--", alpha=0.4)
    plt.xlabel("t")
    plt.ylabel("cumulative reward")
    plt.title(f"{title}: reward mismatch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print trace
    print("\nTrace:")
    for t, (a, b, an) in enumerate(zip(agent.action_history, agent.b_action_history, agent.anomaly_history)):
        print(
            f"t={t:02d} | A={a:>5} | B={b:>10} | "
            f"behavior_anom={an['behavior']} | reward_anom={an['reward']} | "
            f"ruled_out={an['ruled_out_now']} | OOB={an['oob_switch']}"
        )


# -----------------------------
# Demo
# -----------------------------
benign_run = run_episode(BenignB(), T=10, delta=0.12, omega=2.0)
adv_run = run_episode(AdversarialB(switch_time=4), T=10, delta=0.12, omega=2.0)

plot_run(benign_run, title="Benign B")
plot_run(adv_run, title="Adversarial B")