# -*- coding: utf-8 -*-
"""
Classes for all Halotools-specific exceptions.
"""

__all__ = ('HalotoolsError', 'HalotoolsCacheError',
        'HalotoolsIOError', 'UnsupportedSimError', 'CatalogTypeError',
        'HalotoolsModelInputError', 'HalotoolsArgumentError')


class HalotoolsError(Exception):
    """ Base class of all Halotools-specific exceptions.
    """

    def __init__(self, message):
        super(HalotoolsError, self).__init__(message)


class InvalidCacheLogEntry(Exception):
    """ Base class of all Halotools-specific exceptions.
    """

    def __init__(self, message):
        super(InvalidCacheLogEntry, self).__init__(message)


########################################


class HalotoolsCacheError(HalotoolsError):
    """ Custom exception used to indicate that there has been an incorrect attempt to load data from the cache directory into memory.
    """

    def __init__(self, message):
        super(HalotoolsCacheError, self).__init__(message)


class HalotoolsIOError(HalotoolsError):
    """ Catch-all custom exception for incorrect Halotools-specific I/O.
    """

    def __init__(self, message):
        super(HalotoolsIOError, self).__init__(message)


class UnsupportedSimError(HalotoolsCacheError):
    """ Custom exception that is raised when there is an attempt to load a halo catalog into memory that is not recogized by Halotools.
    """

    def __init__(self, simname):

        message = ("\nThe input simname " + simname + " is not recognized by Halotools.\n")

        super(UnsupportedSimError, self).__init__(message)


class CatalogTypeError(HalotoolsCacheError):
    """ Custom exception that is raised when an unrecognized type of data catalog is attempted to be loaded into memory.
    """

    def __init__(self, catalog_type):

        message = "\nInput catalog_type = ``"+catalog_type+"``\n Must be either 'raw_halos', 'halos', or 'particles'.\n"

        super(CatalogTypeError, self).__init__(message)


########################################

class HalotoolsModelInputError(HalotoolsError):
    """ Catch-all custom exception for when a model object method was called without the correct arguments.
    """

    def __init__(self, function_name):
        message = ("\nMust pass one of the following keyword arguments to %s:\n"
        "``halo_table`` or  ``prim_haloprop``" % function_name)
        super(HalotoolsModelInputError, self).__init__(message)


class HalotoolsArgumentError(HalotoolsError):
    """ Catch-all custom exception for instantiating a Halotools-defined class without passing the correct arguments to the constructor.
    """

    def __init__(self, function_name, required_input_list):
        """
        Parameters
        -----------
        function_name : string

        required_input_list : list
                List of strings
        """
        message = "\nMust pass each of the following keyword arguments to " + function_name + ":\n"
        for required_input in required_input_list:
            message = message + required_input + ', '
        message = message[:-2]
        super(HalotoolsArgumentError, self).__init__(message)


class AmurricaError(HalotoolsError):
    """ Built-in mechanism to prevent ridiculous spelling-choice contributions to the repository.
    """

    def __init__(self, basename, linenum, correct_spelling, offending_spelling):
        message = ("\nOn line number "+str(linenum)+" of the source code file ``"+basename+"``,\n"
                "you have incorrectly spelled the word ``"+correct_spelling+"`` "
                "as ``"+offending_spelling+"``.\n"
                "Contributions to Halotools that support King George are strictly forbidden.\n")
        super(AmurricaError, self).__init__(message)
