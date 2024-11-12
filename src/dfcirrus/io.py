import os
import re
import sys
import yaml
import string
import shutil
import subprocess
import numpy as np
from functools import partial, wraps

# Path
package_dir = os.path.dirname(__file__)
# config_dir = os.path.normpath(os.path.join(package_dir, '../configs'))

# Default configuration path
# default_config = os.path.join(config_dir, './config.yml')

### LOGGING ###
import logging
from astropy.logger import AstropyLogger

class Logger(AstropyLogger):
    def reset(self, level='INFO', to_file=None, overwrite=True):
        """ Reset logger. If to_file is given as a string, the output
        will be stored into a log file. """

        for handler in self.handlers[:]:
            self.removeHandler(handler)
            
        self.setLevel(level)
        
        if isinstance(to_file, str) is False:
            # Set up the stdout handlers
            handler = StreamHandler()
            self.addHandler(handler)
            
        else:
            if os.path.isfile(to_file) & overwrite:
                os.remove(to_file)
                
            # Define file handler and set formatter
            file_handler = logging.FileHandler(to_file)
            msg = '[%(asctime)s] %(levelname)s: %(message)s'
            formatter = logging.Formatter(msg, datefmt='%Y-%m-%d|%H:%M:%S')
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
            
        self.propagate = False
    
class StreamHandler(logging.StreamHandler):
    """ A StreamHandler that logs messages in different colors. """
    def emit(self, record):
        stream_print(record.msg, record.levelno)

def stream_print(msg, levelno=logging.INFO):
    """ Enable colored msg using ANSI escape codes based input levelno. """
    levelname = logging.getLevelName(levelno)
    stream = sys.stdout
    
    if levelno < logging.INFO:
        level_msg = '\x1b[1;30m'+levelname+': '+'\x1b[0m'
    elif levelno < logging.WARNING:
        level_msg = '\x1b[1;32m'+levelname+': '+'\x1b[0m'
    elif levelno < logging.ERROR:
        level_msg = '\x1b[1;31m'+levelname+': '+'\x1b[0m'
    else:
        level_msg = levelname+': '
        stream = sys.stderr
        
    print(f'{level_msg}{msg}', file=stream)

logging.setLoggerClass(Logger)
logger = logging.getLogger('Logger')
logger.reset()

######


def find_keyword_header(header, keyword,
                        default=None, input_val=False, raise_error=False):
    """ Search keyword value in header (converted to float).
        Accept a value by input if keyword is not found. """
        
    try:
        val = float(header[keyword])
     
    except KeyError:
        logger.info(f"Keyname {keyword} missing in the header .")
        
        if input_val:
            try:
                val = float(input(f"Input a value of {keyword} :"))
            except ValueError:
                msg = f"Invalid {keyword} values!"
                logger.error(msg)
                raise ValueError(msg)
        elif default is not None:
            logger.info("Set {} to default value = {}".format(keyword, default))
            val = default
        else:
            if raise_error:
                msg = f"{keyword} must be specified in the header."
                logger.error(msg)
                raise KeyError(msg)
            else:
                return None
            
    return val
    
def load_config(filename):
    """ Read a yaml configuration. """
    
    if not filename.endswith('.yml'):
        msg = f"Table {filename} is not a yaml file. Exit."
        logger.error(msg)
        sys.exit()
    
    with open(filename, 'r') as f:
        try:
            return yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:
            logger.error(err)
            
def check_config_keys(config, func):
    """ List all keynames that are not of the function. """
    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    extra_keys = set(config.keys()).difference(argnames)
    logger.warning("{} in config are not parameters.".format(extra_keys))

def config_kwargs(func, config_file):
    """Wrap keyword arguments from a yaml configuration file."""

    # Load yaml file
    config = load_config(config_file)
    logger.info(f"Loaded configuration file {config_file}")
    
    # Wrap the function
    @wraps(func)
    def wrapper(*args, **kwargs):
        config.update(kwargs)
        return func(*args, **config)

    return wrapper
    
