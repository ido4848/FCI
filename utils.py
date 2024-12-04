import os
import pickle
import logging
import time
import argparse
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

logger = logging.getLogger(__name__)

MAXIMAL_RANDOM_SEED = 2**32 - 1

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def int_tuple_type(strings):
    if strings is None:
        return None
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def int_or_int_tuple_type(strings):
    if strings is None:
        return None
    if "," not in strings:
        return int(strings)
    else:
        return int_tuple_type(strings)
    
def float_tuple_type(strings):
    if strings is None:
        return None
    strings = strings.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

def float_or_float_tuple_type(strings):
    if strings is None:
        return None
    if "," not in strings:
        return float(strings)
    else:
        return float_tuple_type(strings)    

def args_to_arg_string(args, keys=None):
    arg_string = ""
    kwargs = vars(args)
    for k, v in kwargs.items():
        if keys is not None and k not in keys:
            continue
        if v is None:
            continue
        if isinstance(v, list):
            # TODO: what if this param is starting with - or with nothing
            arg_string += " --" + k + " " + " ".join([str(item) for item in v])
        elif isinstance(v, tuple):
            arg_string += " --" + k + " " + "[" + ",".join([str(item) for item in v]) + "]"
        else:
            # TODO: what if this param is starting with - or with nothing
            arg_string += " --" + k + " " + str(v)
    return arg_string

class ArgumentSaver:
    def __init__(self):
        self.arguments = {}

    def add_argument(self, *args, **kwargs):
        arg_0 = args[0]
        arg_name = arg_0[2:] if arg_0.startswith("--") else arg_0[1:] if arg_0.startswith("-") else arg_0
        self.arguments[arg_name] = (args, kwargs)

    def add_to_parser(self, parser):
        for arg_name, (args, kwargs) in self.arguments.items():
            if 'short_name' in kwargs:
                kwargs.pop('short_name')
            parser.add_argument(*args, **kwargs)

    def add_to_parser_as_options(self, parser):
        option_args = []
        short_names = []
        for arg_name, (args, kwargs) in self.arguments.items():
            new_args = list(args)
            new_args[0] = new_args[0]+'_options'
            option_args.append(arg_name+'_options')
            new_kwargs = dict(kwargs)
            if 'nargs' in kwargs:
                new_kwargs.pop('nargs')
            if 'const' in new_kwargs:
                new_kwargs.pop('const')
            if 'action' in new_kwargs:
                if new_kwargs['action'] != AddDefaultInformationAction:
                    raise ValueError(f"arg {arg_name} has action that is not AddDefaultInformationAction")
                new_kwargs.pop('action')
            default_value = new_kwargs.pop('default')
            new_kwargs['nargs'] = '+'
            new_kwargs['default'] = [default_value]
            new_kwargs['action'] = AddDefaultInformationAction
            if 'short_name' in new_kwargs:
                short_names.append(new_kwargs['short_name'])
            else:
                short_names.append(None)

            parser.add_argument(*new_args, **new_kwargs)
        return option_args, short_names

class AddDefaultInformationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

class AddOutFileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

        outfile = os.path.join(values, os.path.basename(values)+'.out')
        setattr(namespace, 'outfile', outfile)

class TeeStdout(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

class TeeStderr(object):
    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stderr = sys.stderr
        sys.stderr = self
    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

class TeeAll(object):
    def __init__(self, name, mode='a'):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.tee_stdout = TeeStdout(name, mode)
        self.tee_stderr = TeeStderr(name, mode)
                
def setup_logger(the_logger, outfile=None):
    the_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')    

    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        fh = logging.FileHandler(outfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        the_logger.addHandler(fh)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        the_logger.addHandler(ch)