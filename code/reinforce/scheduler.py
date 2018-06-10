from code.interfaces import IScheduler, ISchedulerConfig
from agent import REINFORCEAgent


class ConfigREINFORCE(ISchedulerConfig):
    def __init__(self, env_config: EnvironmentConfig, args):
        super(self, ConfigREINFORCE).__init__(env_config, args)
        self.epoch_count = epoch_count # 100
        self.sessions_in_epoch  = sessions_in_epoch # 100
        self.max_steps = max_steps # 5000
        self.save_period = save_period
        self.checkpoint = checkpoint


class REINFORCETrainScheduler(IScheduler):

    def __init__(self, config: ConfigREINFORCE, num_frames=4):
        super(self, REINFORCETrainScheduler).__init__(config)
        self.num_frames = num_frames
        self.agent = REINFORCEAgent(image_data, oracle_data,
                traindir_path=args.traindir,
                checkpoint_path=args.checkpoint)

    def generate_session(self, t_max=2000):
        """
        Play a full session with REINFORCE network_agent
        and prepare for train at the session end.
        Returns sequences of states, actions and rewards.
        """

        # Arrays to record session
        states,actions,rewards = [],[],[]
        s = env.reset()
        for t in range(t_max):
            #action probabilities array aka pi(a|s)
            action_probas = predict_proba(np.array([s]))[0]
            a = np.random.choice(n_actions, p=action_probas)
            new_s, r, done, info = env.step(a)

            #record session history to train later
            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = new_s
            if done: break
        return states, actions, rewards

    def get_cumulative_rewards(rewards, #rewards at each step
            gamma = 0.99 #discount for reward
            ):
        G = [rewards[-1]]
        for r in rewards[-2::-1]:
            G.append(r + gamma * G[-1])
        return G[::-1]

    def evaluate_problem(self):
        max_steps = config[RootConfigKeys.MAX_STEPS]
        epoch_count = config[RootConfigKeys.EPOCH_COUNT]
        for ind in range(epoch_count):
            self.generate_session(t_max=max_steps)

    def process_problem(self):
        epoch_count = config[RootConfigKeys.EPOCH_COUNT]
        max_steps   = config[RootConfigKeys.MAX_STEPS]
        save_period = config[RootConfigKeys.SAVE_PERIOD]
        sessions_in_epoch = config.sessions_in_epoch
        for epoch_id in range(epoch_count):
            reward_list = []
            for session_id in sessions_in_epoch:
                states, actions, rewards = self.generate_session(
                    t_max=max_steps)
                sum_rewards = self.agent.train_on_session(
                        states, actions, rewards)
                reward_list.append(sum_rewards)
                if (ind + 1) % save_period == 0:
                    # TODO(nekrald)
                    # Save the network
                    pass
            mean_reward = np.mean(reward_list)

