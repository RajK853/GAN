import yaml
import numpy as np


def load_mnist_data():
    from tensorflow.keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # Make sure images have shape (28, 28, 1)
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    # Scale images to the [0, 1] range
    train_x = train_x.astype("float32")/255.0
    test_x = test_x.astype("float32")/255.0
    # Reshape labels
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    print(f"# Train data shape: {train_x.shape}.")
    print(f"# Train label shape: {train_y.shape}.")
    return (train_x, train_y), (test_x, test_y)


def load_yaml(file_path, safe_load=True):
    """
    Loads a YAML file from the given path
    :param file_path: (str) YAML file path
    :param safe_load: (bool) If True, uses yaml.safe_load() instead of yaml.load()
    :returns: (dict) Loaded YAML file as a dictionary
    """
    load_func = yaml.safe_load if safe_load else yaml.load
    with open(file_path, "r") as fp:        
        return load_func(fp)


def exec_from_yaml(config_path, exec_func, title="Experiment", safe_load=True, ignore_exp_prefix="definition"):
    """
    Executes the given function by loading parameters from a YAML file with given structure:
    Experiment 1 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    Experiment 2 Name:
        argument_1: value_1
        argument_2: value_2
        ...
    NOTE: The argument names in the YAML file should match the argument names of the given execution function.
    :param config_path: (str) YAML file path
    :param exec_func: (callable) Function to execute with the loaded parameters
    :param title: (str) Label for each experiment
    :param safe_load: (bool) If True, uses yaml.safe_load method() instead of yaml.load() to read YAML config file.
    :param ignore_exp_prefix: (str) Experiment name prefix used to prevent certain parameters from being executed
    :returns: (dict) Dictionary with results received from each experiment execution
    """
    result_dict = {}
    config_dict = load_yaml(config_path, safe_load=safe_load)
    i = 1
    for exp_name, exp_kwargs in config_dict.items():
        if exp_name.startswith(ignore_exp_prefix):
            continue
        print(f"\n{i}. {title}: {exp_name}")
        result = exec_func(**exp_kwargs)
        result_dict[exp_name] = result
        i += 1
    return result_dict
