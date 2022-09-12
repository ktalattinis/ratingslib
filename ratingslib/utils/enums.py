"""
This module includes implementation of Enums and the 
available versions of rating systems
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

from typing import Set

from ratingslib.utils.validation import ValidationError


class MyEnum():
    """Class for Enums"""
    @classmethod
    def add(cls, var_name, value):
        attr_name_set = cls.get_attrs_names()
        attr_value_set = cls.get_attrs_values()
        if var_name in attr_name_set:
            raise ValidationError(
                'Variable name exists! Existing names are: [' +
                ','.join(attr for attr in attr_name_set)+']')
        elif value in attr_value_set:
            raise ValidationError(
                'Variable value exists! Existing values are:' +
                '[' + ','.join(val for val in attr_value_set) + ']')
        else:
            setattr(cls, var_name, value)

    @classmethod
    def get_attrs_names(cls) -> Set:
        attr_name_set = set(i for i in cls.__dict__.keys()
                            if i[:1] != '_')
        return attr_name_set

    @classmethod
    def get_attrs_values(cls) -> Set:
        attr_value_set = set(v for i, v in cls.__dict__.items()
                             if i[:1] != '_')
        return attr_value_set


class ratings(MyEnum):
    """Rating Systems versions"""
    COLLEY = 'Colley'
    MASSEY = 'Massey'
    KEENER = 'Keener'
    MARKOV = 'Markov'
    OD = 'OffenseDefense'
    ELOWIN = 'EloWin'
    ELOPOINT = 'EloPoint'
    WINLOSS = 'Winloss'
    ACCURATE = 'AccuRATE'
    # AGGREGATION RATING MODELS
    AGGREGATIONOD = 'AggrOD'
    AGGREGATIONMARKOV = 'AggrMarkov'
    AGGREGATIONPERRON = 'AggrPerron'
    # AGGREGATION RANKING METHODS
    RANKINGBORDA = 'Borda'
    RANKINGAVG = 'AvgRank'
