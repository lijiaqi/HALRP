import os
import errno
import os.path as osp

__all__ = ["mkdir_if_missing", "check_file_exist"]


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def check_file_exist(filename):
    if not osp.isfile(filename):
        raise FileNotFoundError(
            'file "{}" does not exist'.format(osp.abspath(osp.expanduser(filename)))
        )
