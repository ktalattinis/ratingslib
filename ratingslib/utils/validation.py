"""
This module gathers utilities for input validation.
All functions that starts with `validate` raises an error if the input
is not valid.
"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import numbers
from typing import Any, List, Set, Tuple, Union


class ValidationError(Exception):
    """Class to raise a message when there's a validation error"""


def validate_from_set(input_to_check: Any,
                      set_of_values: Set[Any],
                      var_name: str = ""):
    """Check if target value is an element of set

    Parameters
    ----------
    input_to_check : Any
        Target input.

    set_of_values : Set[Any]
        Set data structure.

    var_name : str, default=""
        Variable name is printed in the Error message.

    Raises
    ------
    ValidationError
        If the target value is not an element of target set data structure.
    """
    if input_to_check not in set_of_values:
        raise ValidationError('The parameter "' + var_name + '" with value "' +
                              input_to_check + '" is not valid.' +
                              "\nAccepted values are: " +
                              " or ".join('"%s"' % i for i in set_of_values))


def validate_type(variable: Any, var_type, var_name: str = ""):
    """Checks that the target variable is an instance of var_type

    Parameters
    ----------
    variable : Any,
        Target value.

    var_type : A class, type or a tuple containing classes,\
    types or other tuples
        Target type.

    var_name : str, default=""
        Variable name is printed in the Error message.

    Raises
    ------
    TypeError
        If ``variable`` is not an instance of ``var_type``.
    """
    if not isinstance(variable, var_type):
        raise TypeError(var_name +
                        " parameter must be " +
                        str(var_type.__name__) +
                        " type. " +
                        str(variable) +
                        " is " +
                        str(type(variable).__name__))


def validate_not_none_and_type(variable: Any,
                               var_type,
                               var_name: str = ""):
    """Checks that the target variable is not None and it is an instance of
    var_type.

    Parameters
    ----------
    variable : Any,
        Target value

    var_type : A class, type or a tuple containing classes,\
    types or other tuples
        Target type

    var_name : str, default=""
        Variable name is printed in the Error message.

    Raises
    ------
    TypeError
        If ``variable`` is None.
    """
    if variable is None:
        raise TypeError(var_name + " must not be None")
    validate_type(variable, var_type, var_name)


def validate_type_of_elements(elements: Union[List, Tuple, Set],
                              class_name):
    """Check if the ``elements`` parameter contains objects of same type with
    the target class (parameter ``class_name`` ).
    The parameter ``elements`` must be one of the following data structures:
    List, Tuple or Set

    Parameters
    ----------
    elements : Union[List, Tuple, Set]
        Target list or tuple or set

    class_name : A class, type or a tuple containing classes,\
    types or other tuples
        Target class

    Raises
    ------
    TypeError
        If ``elements`` is not list or tuple or set.
        If the objects of ``elements`` do not have the same type.

    """
    if (isinstance(elements, list) or
        isinstance(elements, tuple) or
       isinstance(elements, set)):
        if not all(isinstance(v, class_name) for v in elements):
            raise TypeError('All elements must be objects of '
                            + str(class_name.__name__) + " class.")
    else:
        raise TypeError('Parameter elements must be a list or tuple or set.')


# ==============================================================================
# Helper functions for check input without raising Error
# ==============================================================================

def is_number(x):
    """
    Return ``True`` if the given parameter is a number

    Examples
    --------
    >>> from ratingslib.utils.methods import is_number
    >>> x = 1000.12324125345678
    >>> is_number(x)
    True
    >>> x = 'A'
    >>> is_number(x)
    False
    """
    return isinstance(x, numbers.Number)


def list_has_only_numbers(list_to_check):
    """Return ``True`` if the given list contains numbers only

    Examples
    --------
    >>> from ratingslib.utils.methods import list_has_only_numbers
    >>> x = [1, 2 , 1000, 1, 'A']
    >>> list_has_only_numbers(x)
    False
    >>> x = [1, 2 , 1000, 1]
    >>> list_has_only_numbers(x)
    >>> True
    """
    return all([is_number(x) for x in list_to_check])
