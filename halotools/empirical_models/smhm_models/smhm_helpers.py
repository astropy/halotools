from ...custom_exceptions import HalotoolsError
from ...sim_manager import sim_defaults
from warnings import warn

unspecified_redshift_warning_msg = ("\nYou did not specify a redshift."
    "Returning results for z = %.2f. \n"
    "You can either pass a ``redshift`` keyword argument to the function, \n"
    "and/or change the default behavior by modifying the value of sim_defaults.default_redshift.\n")

inconsistent_redshift_msg = ("\nYou passed a redshift = %.2f argument to the constructor of the "
    "``%s`` class.\nThis choice implies that your instance of ``%s`` can only be used \n"
    "to study the stellar-to-halo-mass relation at this redshift. "
    "\nHowever, you just passed a redshift = %.2f to the ``%s`` method, which is inconsistent.\n"
    "Depending on what you are trying to do, you should do one of the following:\n"
    "\n1. instantiate the ``%s`` class again without passing in a redshift, \n"
    "which will allow you to pass in whatever redshift you wish to the ``%s`` method,\n\n"
    "2. only pass in a redshift to the ``%s`` method that is consistent with \n"
    "the redshift you passed to the constructor,\n\n"
    "3. do not pass in any redshift argument at all to the ``%s`` method,\n"
    "in which case the redshift you passed to the constructor will be automatically chosen.\n\n"
                             )


def get_inconsistent_redshift_msg(input_redshift, constructor_redshift, class_name, method_name):
    return (inconsistent_redshift_msg %
        (constructor_redshift, class_name, class_name, input_redshift, method_name,
            class_name, method_name, method_name, method_name))


def safely_retrieve_redshift(obj, method_name, **kwargs):
    """
    """
    if 'redshift' in kwargs:
        redshift = kwargs['redshift']
        if hasattr(obj, 'redshift'):
            if redshift != obj.redshift:
                class_name = obj.__class__.__name__
                msg = get_inconsistent_redshift_msg(
                    redshift, obj.redshift, class_name, method_name)
                raise HalotoolsError(msg)
    else:
        if hasattr(obj, 'redshift'):
            redshift = obj.redshift
        else:
            redshift = sim_defaults.default_redshift
            warn(unspecified_redshift_warning_msg % redshift)

    return redshift
