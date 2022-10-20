#!/bin/bash

NOTEBOOK_CONFIG=~/.jupyter/jupyter_notebook_config.py

# backup existing config, is any
cp $NOTEBOOK_CONFIG ~/jupyter_notebook_config_backup.py

mkdir ~/.jupyter
echo "c = get_config();c.NotebookApp.password = u'"$(python -c 'from IPython.lib import passwd;print(passwd())')"'" > $NOTEBOOK_CONFIG

