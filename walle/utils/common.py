import datetime
import os
import shutil


def makedir(dirname):
  """Safely creates a new directory.
  """
  if not os.path.exists(dirname):
    os.makedirs(dirname)


def rmdir(dirname):
  """Deletes a non-empty directory.
  """
  answer = ""
  while answer not in ["y", "n"]:
    answer = input("Permanently delete {} [Y/N]?".format(dirname)).lower()
    if answer == "y":
      shutil.rmtree(filedir, ignore_errors=True)
    else:
      return


def gen_timestamp():
  """Generates a timestamp in YYYY-MM-DD-hh-mm-ss format.
  """
  date = str(datetime.datetime.now()).split('.')[0]
  return date.split(' ')[0] + '-' + '-'.join(date.split(' ')[1].split(':'))


def gen_checkerboard(n, s):
  """Creates an nxn checkerboard of size sxs.
  """
  row_even = (n // 2) * [0, 1]
  row_odd = (n // 2) * [1, 0]
  checkerboard = np.row_stack((n//2)*(row_even, row_odd))
  return checkerboard.repeat(s, axis=0).repeat(s, axis=1)