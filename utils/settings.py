import os
import yaml
import utils.logger as logger
import numbers


class Param:
    """Wrapper for parameter. This stores the value."""

    def __init__(self, value, min_value=None, max_value=None):
        """Constructor for parameter.
        :param value: Can be anything.
        :type value: Can be anything.
        """
        self.value = value
        if not isinstance(value, numbers.Number) and (min_value is not None or max_value is not None):
            logger.error('Supplied a max or min value for a non-numerical parameter')

    def set_to(self, value):
        """Sets the value of the parameter.

        :param value: The value to set the parameter to.

        :type value: Any kind of type.

        :raises TypeError: If the passed value type does not match the existing type. This is only checked for non-
            numerical values.
        """
        if not isinstance(value, numbers.Number) and not isinstance(self.value, numbers.Number):
            if type(value) is not type(self.value):
                raise TypeError("Object types do not match.")

        self.value = value

    def get(self):
        """Returns the value."""
        return self.value

    def type(self):
        """Returns the type of the parameter."""
        return type(self.value)


class Config:
    """ This class stores parameters and saves them at exit."""

    def __init__(self, name: str, root: str, directory: str =''):
        """Initializer for config, creates directories and loads any pre-existing configuration files associated with
        the name.

        :param str name: The name of the config file, also the name of the produced .yaml file.
        :param str root: The root directory for the configuration.
        :param str directory: The directory where the config file should be stored. Set to ".//config//" by default

        :raises OSError: If it cannot create a directory.
        """
        # Test for string
        if not isinstance(directory, str):
            raise ValueError("Directory is not a string")

        if not isinstance(name, str):
            raise ValueError("Config name must be a string.")

        self.name = name

        self.root_dir = root

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

        if len(directory) > 1:
            self.directory = self.root_dir + "/" + directory + "/"
        else:
            self.directory = self.root_dir + "/config/"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            if not os.path.exists(self.directory):
                raise OSError("Could not create directory: " + self.directory)

        self.filename = self.directory + self.name + ".yaml"

        self.keys_added = []

        if not os.path.exists(self.filename):
            logger.info("Could not find pre-existing configuration [{}], using defaults.".format(name))
            self.params = {}
        else:
            logger.info("Found existing configuration [{}], loading values.".format(name))
            self.load()

        pass

    def add_param(self, param_name: str, value) -> Param:
        """Prototype to add a param by value, returns the parameter itself.

        :param str param_name: The parameter name, which is used as a key for retrieval.
        :param value: The initial value of the parameter. This is only used if the parameter is not loaded
            from the existing file, in which case the stored value is used.

        :type value: Can be anything.
        """
        add = False
        if self.params is None:
            self.params = {}
            add = True
        elif param_name not in self.params:
            add = True

        if add:
            self.params[param_name] = Param(value)

        self.keys_added.append(param_name)

        return self.params[param_name]

    def get(self, param_name: str) -> Param:
        """Returns the parameter by name using dict."""
        return self.params[param_name]

    def get_root_dir(self):
        return self.root_dir

    def load(self):
        """Loads the configuration from disk and populates the param dict with the values.

        :raises OSError: If file cannot be read/found.
        """

        try:
            with open(self.filename, 'r') as config_file:
                serialized = config_file.read()
                if len(serialized) < 10:
                    self.params = {}
                else:
                    self.params = yaml.load(serialized)

        except OSError:
            logger.error("Error encountered reading config file: {0}".format(self.filename))

    def __exit__(self, exc_type, exc_value, traceback):
        """Saves the file on exit."""
        self.save()

    def __delete__(self, instance):
        self.save()

    def save(self):
        """Serializes and saves the parameter dict to disk using YAML.

        :raises OSError: If unable to save file to disk.
        """
        keys_to_remove = []

        # Cleanup un-used params
        for key in self.params:
            if key not in self.keys_added:
                keys_to_remove.append(key)

        for value in keys_to_remove:
            del self.params[value]

        serialized_str = yaml.dump(self.params)

        try:
            with open(self.filename, 'w') as config_file:
                config_file.write(serialized_str)
        except OSError:
            logger.error("Error encountered saving config file: {0}".format(self.filename))
