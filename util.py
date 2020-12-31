import random
import numpy as np


def decision(probability):
    """
    :param probability: number between 0-1
    :return: True or False
    """
    return random.random() < probability


def create_dict_of_counts(tuple_list):
    """
    :param tuple_list: list of tuples
    :return: dict of how many times that tuple is in list
    """
    new_dict = dict()
    for pair in tuple_list:
        try:
            count = new_dict[pair]
            new_dict.update({pair: count + 1})
        except KeyError:
            new_dict.update({pair: 1})
    return new_dict


def dict_of_array_values_to_lists(dictionary):
    """
    :param dictionary: dict, where values are numpy arrays
    :return: dict, where values are lists
    """
    new_dict = dict()
    for key in dictionary:
        new_dict.update({key: dictionary[key].tolist()})
    return new_dict


def dict_of_list_values_to_arrays(dictionary):
    """
    :param dictionary: dict, where values are lists
    :return: dict, where values are numpy arrays
    """
    new_dict = dict()
    for key in dictionary:
        new_dict.update({key: np.array(dictionary[key])})
    return new_dict
