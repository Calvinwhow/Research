import os
import yaml
import warnings

NIMLAB_CONFIG_PATH = os.path.join(os.getenv("HOME"), "setup/nimlab_config.yaml")

if os.path.exists(NIMLAB_CONFIG_PATH):
    with open(NIMLAB_CONFIG_PATH) as yml:
        config = yaml.safe_load(yml)
else:
    raise FileNotFoundError(
        f"Cluster config file not found! Please create a config file at ~/setup/nimlab_config.yaml"
    )

software = config["software"]
database = config["database"]
volume_spaces = config["volume_spaces"]
surface_spaces = config["surface_spaces"]
v2s_compatible_conn = config["v2s_compatible_conn"]
s2v_compatible_conn = config["s2v_compatible_conn"]
connectomes = config["connectomes"]
clusters = config["clusters"]


def verify_software(software_list, ignore=False):
    """Verify if software paths are configured.

    Parameters
    ----------
    software_name : list
        List of software paths to verify.
    """
    for tool in software_list:
        if software[tool] == "":
            if ignore:
                warnings.warn(f"{tool} is not set in config file", RuntimeWarning)
            else:
                raise RuntimeError(f"{tool} is not set in config file")
