# Dynamix

## Description
Dynamix is a novel system designed to automate DJ functions by dynamically interpreting the affective state of a crowd in real time. Utilizing advanced affect recognition models, Dynamix enhances party atmospheres by optimizing music selection and transitions based on crowd emotion, aiming to maintain and elevate the overall energy of the venue.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Folder Structure](#folder-structure)
4. [Dynamix App](#dynamix-app)


## Installation
To install Dynamix, follow these steps:
1. Clone the repository to your local machine. Open your terminal or command prompt, navigate to the directory where you want to clone the repository, and run the following command: `$ git clone https://github.com/liamkronman/dynamix.git`

2. Install the required dependencies using pip. Assuming you have Python and pip installed, you can install the dependencies listed in the requirements.txt file by running the following command: `$ pip install -r requirements.txt`


## Usage
To start up the system, follow these steps:

1. Open your terminal or command prompt.

2. Navigate to the 'affect' directory by running the following command:

```bash
$ cd affect
```

3. Once you're in the 'affect' directory, run the Python script 'test_integration.py' by executing the following command: `$ python test_integration.py`


## Folder Structure
The organization of this repository is as follows:
```
/
│
├── affect/             # Folder for streaming, frame capturing and sentiment analysis logic 
│   ├── test_integration.py # The file to run the system!
│   └── ...
│
├── dynamix-app/        # Folder for Dynamix app that streams video with client and plays audio in the backend
│   ├── backend/    
│   └── frontend      
│   
│
├── songs/              # Contains songs used in the system
│
├── spotify/            # Folder for scripts that use Spotify's API to analyze songs for audio features
│   ├── spotify/good_spotify_playlist_features.csv  # CSV for storing audio feature information for each song
│   └── ...
│
└── transitions/        # Folder containing helper functions for transitioning between songs
    └── ...
```

## Dynamix App
The Dynamix app is still a WIP and it is preferable that you follow the instructions in the Usage (#usage) section instead to run the system. However if you would like to check it out:

1. Navigate to the `dynamix-app/frontend` directory by running the following command from the root of the cloned repository:

```bash
$ cd dynamix-app/frontend
```

2. Once you're in the dynamix-app/frontend directory, install the necessary dependencies by running:
```bash
$ npm install
```

3. After the installation is complete, start the frontend server by running:
``` bash
$ npm start
```

4. In a new terminal, navigate to the `dynamix-app/backend` directory by running the following command from the root of the cloned repository:
```bash
$ cd dynamix-app/backend
```

5. Run the backend script via the following command:
```bash
$ python backend.py
```
