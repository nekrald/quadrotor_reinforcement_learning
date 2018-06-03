from code.interfaces import IScheduler
from agent import REINFORCEAgent

class REINFORCETrainScheduler(IScheduler):

    def __init__(self, config, args):
        super(self, REINFORCETrainScheduler).__init__(config, args)
        self.size_rows =
        self.size_cols =
        self.num_frames =

    def generate_session(self, t_max):
        states, actions, rewards = [], [], []
        self.reward_processor.reset()
        self.action_processor.reset()
        self.reset_client()

        current_state =
        for tm in range(t_max):
            action_probas = self.agent.predict_proba(
                    np.array([current_state]))[0]
            action = np.random.choice(n_actions, p=action_probas)
            quad_offset = action_processor.interpret_action(action)
            quad_before_state = client.getPosition()
            if args.forward_only:
                if len(quad_offset) == 1:
                    client.rotateByYawRate(quad_offset[0],
                            move_duration)
                else:
                    client.moveByVelocity(
                        quad_offset[0], quad_offset[1],
                        quad_offset[2], move_duration,
                        DrivetrainType.ForwardOnly)
            else:
                client.moveByVelocity(
                    quad_offset[0], quad_offset[1],
                    quad_offset[2], move_duration,
                    DrivetrainType.MaxDegreeOfFreedom)
            quad_position = client.getPosition()
            quad_velocity = client.getVelocity()
            collision_info = client.getCollisionInfo()

            new_state =

            states.append(current_state)
            actions.append(action)
            rewards.append(reward)

            current_state = new_state
            done =
            if done: break
        return states, actions, rewards


    def reset_client(self):
        logging.info("Resetting the client.")
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        if not self.config[RootConfigKeys.USE_FLAG_POS]:
            self.client.simSetPose(Pose(Vector3r(
                self.initX, self.initY, self.initZ),
                AirSimClientBase.toQuaternion(0, 0, 0)),
                ignore_collison=True)

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

        self.reward_processor = make_reward(self.config, self.client)
        self.action_processor = make_action(self.config)

        self.agent = REINFORCEAgent(image_data, oracle_data,
                traindir_path=args.traindir,
                checkpoint_path=args.checkpoint)

        launch_reward_sum = 0
        while current_step < max_steps:
            action = agent.act(current_state)



            agent.observe(current_state, action, reward, done)
            agent.train()








