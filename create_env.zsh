#!/bin/zsh

# Copied conda initialize script from conda init zsh 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/homebrew/Caskroom/miniconda/base/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
        . "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
    else
        export PATH="/opt/homebrew/Caskroom/miniconda/base/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# This script will create env if it does not exist. If env exists it will delete and recreate the environment

environment_name="discovery_env"
# --------------------------------------------------
# 1. Check if current environment is $discovery_env and deactivate
# --------------------------------------------------
if [[ $CONDA_DEFAULT_ENV == $environment_name ]]; then
    conda deactivate
    echo "Conda environment $environment_name deactivated"
fi

# --------------------------------------------------
# 2. Deletes conda environment if it exists
# --------------------------------------------------
if conda env list | grep -q "\<$environment_name\>"; then
    echo "Conda environment $environment_name exists. Removing it"
    conda env remove -n "$environment_name"
    echo "Conda environment $environment_name has been removed."
fi


# --------------------------------------------------
# 3. Creates conda environment from dev_environment.yml
# --------------------------------------------------
# check if its intel mac and use the right environment file
if [[ $(uname -m) == "arm64" ]]; then
    echo "Creating conda environment for Apple Silicon"
    conda env create -f dev_environment.yml
else
    echo "Creating conda environment for Intel Mac"
    conda env create -f dev_intel_mac_environment.yml
fi
echo "Conda environment $environment_name has been created."

# --------------------------------------------------
# 4. Activates conda environment
# --------------------------------------------------
conda activate $environment_name
echo "Conda environment $environment_name activated."
