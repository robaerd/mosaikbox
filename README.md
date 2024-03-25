# Mosaikbox

This repository contains the backend of Mosaikbox, an automatic music mixing system.

Besides the backend, this repository also contains the survey implementation for the user study conducted in the context of this project.

The frontend of Mosaikbox is available at [mosaikbox-frontend](github.com/...).


## How to run the app

### Configuration
The configuration is done in `app/config/config.py`. Following configuration needs to be done before running the app for the first time:
- Set the path where Mosaikbox stores its data by changing `COMPUTATION_PATH_DEV` and `COMPUTATION_PATH_PROD`, for local and production environments, respectively.
- Change the `API_KEY` of the `https://genius.com` API.

### GPU
The app has been developed on a Macbook and thus has the `mps` GPU device hardcoded in the code for the `torch` models.
If you want to run the app on another GPU you will have to adopt the device type to your system at the following locations: `music_source_separation.py` and `beats.py`.
We strongly recommend using a GPU and discourage from running the app on a CPU, as the computation time is significantly longer.

### Dependencies
- Python 3.11 (should work with 3.9+, at the time of writing some packages are not yet compatible with 3.12)
- Docker (for running the KeyFinderService and MongoDB for the survey)

### Install
```sh
python3.11 -m venv ./python_3.11_venv.nosync

source python_3.11_venv.nosync/bin/activate  # activate venv

pip install -r requirements.txt
```

### Run
```sh
docker-compose up -d # start KeyFinderService and the MongoDB for the survey
uvicorn app.main:app
```

## Frontend
The frontend is available at [mosaikbox-frontend](github.com/...)
