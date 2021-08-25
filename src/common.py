import constants
import json


def is_segmentation_task(task):
    return is_single_segmentation_task(task)


def is_single_segmentation_task(task):
    if task == constants.TASK_SEG:
        return True
    else:
        return False


def is_sequence_labeling_task(task):
    return is_segmentation_task(task)


def use_fmeasure(keys):
    for key in keys:
        if key.startswith('B-'):
            return True

    return False


def has_attr(var, attr):
    return hasattr(var, attr)


def get_jsons(fd):
    # str to dictionary (json-like)
    ret = json.loads(fd)
    return ret


def get_json(fd):
    # json to dictionary
    ret = json.load(fd)
    return ret


def get_concat_strings(x, y):
    assert isinstance(x, str) and isinstance(y, str)
    ret = x + y
    return ret


def get_dict_by_indexes(dic, indexes):
    return {index: dic[index] for index in indexes}
