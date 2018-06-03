
from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase

class IScheduler(object):

    def __init__(self, config, args):

        # Copy initialization to fields.
        self.config = config
        self.args = args

        # Activate client.
        self.client = MultirotorClient()
        self.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Initial position moves.
        initial_position = self.client.getPosition()
        if config[RootConfigKeys.USE_FLAG_POS]:
            self.initX = initial_position.x_val
            self.initY = initial_position.y_val
            self.initZ = initial_position.z_val
            logging.info("Staying at initial coordinates:"
                + "({}, {}, {})".format(
                    self.initX, self.initY, self.initZ))
        else:
            self.initX = config[RootConfigKeys.INIT_X]
            self.initY = config[RootConfigKeys.INIT_Y]
            self.initZ = config[RootConfigKeys.INIT_Z]
            logging.info(
                ("Ignoring flag. Using coordinates (X, Y, Z):{}"
                + ", Rotation:{}").format(
                    (self.initX, self.initY, self.initZ), (0, 0, 0)))
            self.client.simSetPose(Pose(Vector3r(
                self.initX, self.initY, self.initZ),
                AirSimClientBase.toQuaternion(0, 0, 0)),
                ignore_collison=True)

    def process_problem(self):
        raise NotImplementedError("Abstract class!")
