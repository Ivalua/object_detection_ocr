import sys
import os
import re

def check_config(write_json=False):
    import json
    with open(os.path.expanduser('~') + '/.keras/keras.json') as data_file:
        data = json.load(data_file)
    backend = data["backend"]

    r = re.search('/envs/(cntk|keras-tf)-py([0-9])([0-9])/bin/python', sys.executable)
    conda_env = { "tensorflow" : "keras-tf" , "cntk": "cntk" }

    if backend not in conda_env:
        sys.exit("Backend not supported.")
    else:
        env_name = conda_env[backend] + "-py" + str(sys.version_info.major) + str(sys.version_info.minor)

    if r is None or (sys.version_info.major != int(r.group(2)) or sys.version_info.minor != int(r.group(3))):
    	sys.exit("""
To create corresponding environment:

    	conda env update --file """ + env_name + """.yml

To activate Conda environnement

    	source activate """ +  env_name + """

    	""")
    else:
        if conda_env[backend] != r.group(1):
            for b,e in conda_env.items():
                if e == r.group(1):
                    os.environ["KERAS_BACKEND"] = b
                    if write_json:
                        print("Modifying ~/.keras/keras.json to " + b)
                        with open(os.path.expanduser('~') + '/.keras/keras.json', "w") as data_file:
                            json.dump(data, data_file)
