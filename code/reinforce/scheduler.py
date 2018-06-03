from code.interfaces import IScheduler
from agent import REINFORCEAgent

class REINFORCETrainScheduler(IScheduler):

    def __init__(self, config, args):
        super(self, REINFORCETrainScheduler).__init__(config, args)
        self.size_rows =
        self.size_cols =
        self.num_frames =

    def process_problem(self):
        epoch_count = config[RootConfigKeys.EPOCH_COUNT]
        max_steps = epoch_count * config[RootConfigKeys.MAX_STEPS_MUL]
        current_step = 0

        responses = client.simGetImages(
            [ImageRequest(3, AirSimImageType.DepthPerspective,
            True, False)])

        image_state = transform_input(responses)

        # TODO(nekrald):
        #   In case of complete information,
        #   coordinates are also attached to state.


        image_data = (self.num_frames, self.size_rows, self.size_cols)
        oracle_data = None

        reward_processor = make_reward(self.config, self.client)
        action_processor = make_action(self.config)

        agent = REINFORCEAgent(image_data, oracle_data,
                traindir_path=args.traindir,
                checkpoint_path=args.checkpoint)

        while current_step < max_steps:
            action = agent.act(current_state)
            agent.observe(current_state, action, reward, done)
            agent.train()








