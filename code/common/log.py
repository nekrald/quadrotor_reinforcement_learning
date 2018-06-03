import logging
import uuid


def configure_logging(traindir):
    main_file_path = os.path.join(traindir,
        "main." + str(uuid.uuid4()) + ".log")
    logging.basicConfig(level=logging.DEBUG,
	format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=main_file_path,
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

