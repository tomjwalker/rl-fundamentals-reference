import os


def set_filepath(filepath):
    """
    Set the filepath. If any directories do not currently exist in the filepath (which may be nested, e.g. /a/b/c/),
    create them.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    return filepath
