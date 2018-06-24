from code.interfaces import IScheduler, ISchedulerConfig
from agent import REINFORCEAgent


def make_scheduler_config(json_config):
    raise NotImplementedError


def make_scheduler(reinforce_config):
    raise NotImplementedError


class SummaryKeys(object):
    MEAN_SESSION_REWARD   = "mean_reward"
    MINIMAL_SESSION_REWARD = "min_reward"


class ConfigREINFORCE(ISchedulerConfig):

    def __init__(self,
            env_config: EnvironmentConfig,
            epoch_count=100,
            sessions_in_epoch=1000,
            max_steps=3000,
            save_period=2000,
            save_path="traindir",
            summary_file_name="summary.log",
            chekpoint_path=None):
        super(self, ConfigREINFORCE).__init__(env_config)

        self.epoch_count = epoch_count
        self.sessions_in_epoch  = sessions_in_epoch
        self.max_steps = max_steps

        # Checkpointing configuration.
        self.save_period = save_period
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path

        # Summary writing configuration.
        self.summary_period = summary_period


class REINFORCETrainScheduler(IScheduler):

    def __init__(self, config: ConfigREINFORCE):
        super(self, REINFORCETrainScheduler).__init__(config)
        self.epoch_count = config.epoch_count
        self.sessions_in_epoch = config.sessions_in_epoch
        self.max_steps = config.max_steps
        self.save_period = config.save_period
        self.save_path = config.save_path

        self.checkpoint_path = checkpoint_path
        self.frames_in_state = config.frames_in_state

        self._prepare_agent_config()

        self.agent = AgentREINFORCE(self.agent_config)
        self.n_actions = self.agent_config.n_actions
        self.last_frames = []

    def _prepare_agent_config(self):
        self.agent_config = None
        raise NotImplementedError

    def _join_frames(self, next_state):
        raise NotImplementedError

    def generate_session(self, t_max=2000):
        """
        Play a full session with REINFORCE network_agent
        and prepare for train at the session end.
        Returns sequences of states, actions and rewards.
        """

        # Arrays to record session
        states,actions,rewards = [],[],[]
        self.last_frames = []
        s = env.reset()
        for t in range(t_max):
            #action probabilities array aka pi(a|s)
            prepared_state = self._join_frames(s)
            action_probas = predict_proba(prepared_state)[0]
            a = np.random.choice(n_actions, p=action_probas)
            new_s, r, done, info = env.step(a)

            #record session history to train later
            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = new_s
            if done: break
        return states, actions, rewards

    def get_cumulative_rewards(self, rewards, gamma = 0.9999):
        G = [rewards[-1]]
        for r in rewards[-2::-1]:
            G.append(r + gamma * G[-1])
        return G[::-1]

    def evaluate_problem(self):
        max_steps = config[RootConfigKeys.MAX_STEPS]
        epoch_count = config[RootConfigKeys.EPOCH_COUNT]
        for ind in range(epoch_count):
            self.generate_session(t_max=max_steps)

    def process_problem(self, gamma=0.9999):
        epoch_count = config[RootConfigKeys.EPOCH_COUNT]
        max_steps   = config[RootConfigKeys.MAX_STEPS]
        save_period = config[RootConfigKeys.SAVE_PERIOD]
        sessions_in_epoch = config.sessions_in_epoch
        for epoch_id in range(epoch_count):
            reward_list = []
            for session_id in sessions_in_epoch:
                states, actions, rewards = self.generate_session(
                    t_max=max_steps)
                cumulative_rewards = get_cumulative_rewards(
                        rewards, gamma)
                sum_rewards = self.agent.train_on_session(
                        states, actions, rewards)
                reward_list.append(sum_rewards)
                if (ind + 1) % save_period == 0:
                    # TODO(nekrald)
                    # Save the network
                    pass
            mean_reward = np.mean(reward_list)

    def _write_summary(self):
        raise NotImplementedError

