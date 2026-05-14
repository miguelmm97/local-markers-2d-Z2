#%% modules setup

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Managing data
import h5py
import os


# %% Logging setup
def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in Logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel("TRACE", logging.DEBUG - 5)
logger_functions = logging.getLogger('functions')
logger_functions.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(formatter)
logger_functions.addHandler(stream_handler)

# hey
#%% # Managing data
def get_fileID(file_list, common_name='datafile'):
    expID = 0
    for file in file_list:
        if file.startswith(common_name) and file.endswith('.h5'):
            stringID = file.split(common_name)[1].split('.h5')[0]
            ID = int(stringID)
            expID = max(ID, expID)
    return expID + 1

def store_my_data(file, name, data):
    try:
        file.create_dataset(name=name, data=data)
    except Exception as ex:
        logger_functions.warning(f'Failed to write {name} in {file} because of exception: {ex}')

def attr_my_data(dataset, attr_name, attr):
    try:
        dataset.attrs.create(name=attr_name, data=attr)
    except Exception as ex:
        logger_functions.warning(f'Failed to write {attr_name} in {dataset} because of exception: {ex}')

def load_my_data(file_list, directory):
    # Generate a dict with 1st key for filenames, 2nd key for datasets in the files
    data_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        data_dict[file] = {}

        with h5py.File(file_path, 'r') as f:
            # Reading groups from the datafile
            for group in f.keys():
                # Reading subgroups/datasets from the group
                data_dict[file][group] = {}
                for subgroup in f[group].keys():
                    try:
                        # Reading datasets in the subgroup
                        data_dict[file][group][subgroup] = {}
                        for dataset in f[group][subgroup].keys():
                            if isinstance(f[group][subgroup][dataset][()], bytes):
                                data_dict[file][group][subgroup][dataset] = f[group][subgroup][dataset][()].decode()
                            else:
                                data_dict[file][group][subgroup][dataset] = f[group][subgroup][dataset][()]
                    except AttributeError:
                        try:
                            # Reading datasets from the group
                            if isinstance(f[group][subgroup][()], bytes):
                                data_dict[file][group][subgroup] = f[group][subgroup][()].decode()
                            else:
                                data_dict[file][group][subgroup] = f[group][subgroup][()]
                        # Case when there is no group
                        except AttributeError:
                            if isinstance(f[group][()], bytes):
                                data_dict[file][group] = f[group][()].decode()
                            else:
                                data_dict[file][group] = f[group][()]
    return data_dict

def load_my_attr(file_list, directory, dataset):
    attr_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        attr_dict[file] = {}
        print(file)

        with h5py.File(file_path, 'r') as f:
            for att in f[dataset].attrs.keys():
                attr_dict[file][att] = f[dataset].attrs[att]

    return attr_dict

def store_my_dict(file, dict):
    for key in dict.keys():
        try:
            file.create_dataset(name=f'{key}', data=dict[key])
        except Exception as ex:
            logger_functions.warning(f'Failed to write key {key} in {file} because of exception: {ex}')


