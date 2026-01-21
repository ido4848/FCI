import os
import pickle
import logging
import time
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

logger = logging.getLogger(__name__)

class Status:
    @classmethod
    def get_status_from_filename(cls, filename):
        if os.path.exists(filename):
            logger.info("Loading {} status from filename".format(cls))
            with open(filename, "rb") as sf:
                return pickle.load(sf, encoding='latin1')
        else:
            raise Exception("filename not found.")

    @classmethod
    def get_status(cls, args):
        status_file = os.path.join(args.save_folder, "status.pickle")
        logger.info("args.save folder is {}".format(args.save_folder))
        if os.path.exists(args.save_folder):
            if os.path.exists(status_file):
                logger.info("Loading {} status from directory".format(cls))
                with open(status_file, "rb") as sf:
                    return pickle.load(sf, encoding='latin1')
            else:
                logger.info("Initiating new {} status from scratch".format(cls))
                return cls(args)
        else:
            logger.info("Initiating new {} status from scratch".format(cls))
            return cls(args)

    def __init__(self, args):
        self.args = args
        self.kwargs = vars(args)
        self.status_file = os.path.join(self.args.save_folder, "status.pickle")
        self.duration = 0
        self.current_start_time = None

        os.makedirs(self.args.save_folder, exist_ok=True)
        self.save()

    def save(self):
        with open(self.status_file, "wb") as sf:
            pickle.dump(self, sf, protocol=2)
        
    def start(self):
        self.current_start_time = time.time()

    def finish(self, should_save=True):
        if self.current_start_time is None:
            raise Exception("Cannot finish without starting first")

        self.duration += time.time() - self.current_start_time
        if should_save:
            self.save()

        self.current_start_time = None
        return self.duration

    def get_duration(self):
        if self.current_start_time is not None:
            raise Exception("Cannot get duration while running")
        return self.duration
