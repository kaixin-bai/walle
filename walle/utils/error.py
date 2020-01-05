"""Class definitions for various Exceptions.
"""

class Error(Exception):
  """Base class for exceptions in this module.
  """
  def __init__(self, msg):
    print(msg)