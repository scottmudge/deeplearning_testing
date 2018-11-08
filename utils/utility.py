import sys
import os
from threading import Thread, Lock, Event
from utils.logger import get_logger as logging
from pathlib import Path
import shutil
import datetime

DateFormat = '%Y-%m-%d %H:%M:%S'


def get_time_stamp_str() -> str:
    """Returns timestamp as string"""
    return datetime.datetime.now().strftime(DateFormat)


def get_datetime_from_timestamp(timestamp: str) -> datetime.datetime:
    """Returns datetime object from string"""
    return datetime.datetime.strptime(timestamp, DateFormat)


def get_timestamp_from_datetime(dt: datetime.datetime) -> str:
    """Returns string from datetime object."""
    return dt.strftime(DateFormat)


def timedelta_to_string(td: datetime.timedelta) -> str:
    """Returns string from time delta."""
    return str(td)


class MovingAvg:
    """Helps with creating a simple moving average."""

    def __init__(self, frame_size: int):
        """Initializes moving average object.

        Args:
            frame_size (int): The number of averaging frames to use.
        """
        self._sum = 0.0
        self._cur_elem_count = 0
        self._buffer = []
        self._frame_size = frame_size
        self._mtx = Lock()

    def add_value(self, val: float):
        """Adds a value to the buffer.

        Args:
            val (float): Value to add into buffer.
        """
        self._mtx.acquire()
        self._buffer.append(val)
        self._sum += val

        if self._cur_elem_count < self._frame_size:
            self._cur_elem_count += 1
        else:
            self._sum -= self._buffer[0]
            self._buffer.pop(0)
        self._mtx.release()

    def set_frame_fount(self, frame_count: int):
        """Sets the frame count

        Args:
            frame_count (int): Frame count to set averager to.
        """
        self._mtx.acquire()
        self._frame_size = frame_count
        self._buffer.clear()
        self._cur_elem_count = 0
        self._sum = 0.0
        self._mtx.release()

    def get_avg(self) -> float:
        """Returns the average value."""
        if self._cur_elem_count < 1:
            return 0
        self._mtx.acquire()
        avg = self._sum / float(self._cur_elem_count)
        self._mtx.release()
        return avg


SharesDirectory = ""


def get_cur_directory(file_name: str=__file__) -> str:
    """Returns the current directory of the file.

    Args:
        file_name (str):
    """
    if hasattr(sys, 'frozen') and sys.frozen:
        path, filename = os.path.split(sys.executable)
        directory = path
    else:
        directory = os.path.dirname(os.path.realpath(file_name))
    return directory


def get_root_directory() -> str:
    """Returns root directory, useful for storing data."""
    return "{}/../".format(get_cur_directory(__file__))


def file_exists(file_name: str) -> bool:
    """
    Args:
        file_name (str):
    """
    if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
        return True
    else:
        return False


def make_directory(directory: str) -> bool:
    """ Creates supplied directory and checks to make sure it exists.

    Args:
        directory (str): Directory to create.

    Return:
        True on success.
    """
    if not os.path.exists(directory):
        # Try to make directory
        try:
            os.mkdir(str(directory))
        except Exception as e:
            logging("Utility").error("Could not create directory:\n\t> {}\n\t> Exception: {}"
                                     .format(directory, str(e)))
            return False
        if not os.path.exists(directory):
            logging("Utility").error("Could not create directory:\n\t> {}".format(directory))
            return False

    return True


def copy_file_to_shares(source_file: str, dest_folder: str, dest_filename: str = "") -> bool:
    """Copies supplied source file to destination file in shares folder. :param
    source_file: Source file to copy. :type source_file: str :param dest_folder:
    Destination folder to copy to. :type dest_folder: str

    Args:
        source_file (str):
        dest_folder (str):

    Returns:
        True on success.
    """

    # Get file name from source path
    file_name = Path(source_file).name

    # Check if file exists
    if not file_exists(source_file):
        logging("Utility").warn("Supplied file does not exist:\n\t> {}".format(source_file))
        return False

    # Check if shares directory is set
    if SharesDirectory is None or len(SharesDirectory) < 3:
        logging("Utility").error("SharesDirectory is not set! Cannot copy file to shares directory:\n\t> {}"
                                 .format(source_file))
        return False

    # Check if shares directory exists or not
    if not make_directory(SharesDirectory):
        logging("Utility").error("Could not verify shares directory exists:\n\t> {}".format(SharesDirectory))
        return False

    # Get endpoint path using shares directory, destination folder, and file name extracted from source file.
    end_dir = SharesDirectory + "/" + dest_folder
    end_file_path = end_dir + "/" + file_name

    if len(dest_filename) > 3:
        end_file_path = end_dir + "/" + dest_filename

    if not make_directory(end_dir):
        logging("Utility").error("Could not verify that end-point directory exists:\n\t> {}".format(end_dir))
        return False

    try:
        shutil.copyfile(source_file, end_file_path)
    except Exception as e:
        logging("Utility").error("Could not copy file to: \n\t> {} \n\t> Exception: {}"
                                 .format(end_file_path, str(e)))
        return False

    logging("Uility").info("Successfully copied local file to shares.\n\t> Local File: {}\n\t> Remote File: {}".format(
        source_file, end_file_path
    ))

    return True


def set_shares_directory(shares_dir: str):
    """Sets the current shares directory. :param shares_dir: Directory to set
    the shares to. :type shares_dir: str

    Args:
        shares_dir (str):
    """
    global SharesDirectory

    SharesDirectory = shares_dir


class KillableThread(Thread):
    """Wraps a killable thread that loops at a preset interval. Runs supplied
    target function.
    """

    def __init__(self, name, target, sleep_interval: int = 1):
        """
        Args:
            name: Name of the thread, used for logging.
            target (function): Target function
            sleep_interval (int): Sleep interval between loops.
        """
        super().__init__()
        self._trigger = Event()
        self._interval = sleep_interval
        self._target = target
        self._name = name
        self._kill = False

    def trigger(self):
        """Triggers loop, but does not kill it."""
        self._kill = False
        self._trigger.set()

    def run(self):
        """Runs the thread."""
        logging("Thread ({})".format(self._name)).info("Starting thread...")
        while True:
            self._target()

            # If no kill signal is set, sleep for the interval,
            # If kill signal comes in while sleeping, immediately
            #  wake up and handle
            is_triggerer = self._trigger.wait(self._interval)
            if is_triggerer:
                if self._kill:
                    break
                else:
                    logging("Thread ({}) has been triggered!".format(self._name))
                    self._trigger.clear()

        logging("Thread ({})".format(self._name)).info("Killing thread...")

    def kill(self):
        """Kills the thread."""
        self._kill = True
        self._trigger.set()