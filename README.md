# NeuralBases
Code base for my Bachelor Thesis about comparing and evaluating the traditional approach of using Endgame Tablebases for evaluating endgame positions with the approach of using Neural Networks to avoid the use of outside knowledge and deal with the large storage demands of Tablebases.

## Setup
The project should ideally be ran on a virtual environment to avoid conflicts with locally existing versions of the used packages and libraries.

**DISCLAIMER**
The source code depends on having the Syzygy 6 man Tablebase locally, and the [Downloader](https://github.com/Mufasaxi/NeuralBases/blob/main/downloader.py) is responsible for that, but this may take a significant amount of time, and requires somewhere around 70GBs of storage space. 

### Initialising the virtual environment
For this project virtualenv was used, but any other virtual environment should work.
To use virtualenv make sure it is installed:
```
pip install virtualenv
```
From there create a virtual environment in your dersired directory
```
python -m venv env
```

Activating the virtual environment depends on the OS one is using.

Once the virtual environment is activated the required packages and dependencies can be installed using:
```
pip install -r requirements.txt
```

## Source code
In the ```src``` directory one could find the documented files necessary for probing the tablebase. The FEN used in the example are arbitrary.
The [UCI Engine](https://github.com/Mufasaxi/NeuralBases/blob/main/src/uci_prober_engine.py) is there in need of integration with a GUI or for setting up the prober to play with other engines.
