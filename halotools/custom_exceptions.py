"""
Classes for all Halotools-specific exceptions.
"""

__all__ = ('HalotoolsError', 'InvalidCacheLogEntry')


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

class AmurricaError(HalotoolsError):
    """ Built-in mechanism to prevent ridiculous spelling-choice contributions to the repository.
    """

    def __init__(self, basename, linenum, correct_spelling, offending_spelling):
        message = ("\nOn line number "+str(linenum)+" of the source code file ``"+basename+"``,\n"
                "you have incorrectly spelled the word ``"+correct_spelling+"`` "
                "as ``"+offending_spelling+"``.\n"
                "Contributions to Halotools that support King George are strictly forbidden.\n")
        super(AmurricaError, self).__init__(message)
