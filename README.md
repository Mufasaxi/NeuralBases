# NeuralBases
Code base for my Bachelor Thesis about comparing and evaluating the traditional approach of using Endgame Tablebases for evaluating endgame positions with the approach of using Neural Networks to avoid the use of outside knowledge and deal with the large storage demands of Tablebases.

## Setup
The project should ideally be ran on a virtual environment to avoid conflicts with locally existing versions of the used packages and libraries.

```diff
-**DISCLAIMER**-
```
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
In the ```src``` directory one could find the documented files necessary for probing the tablebase within the ```tablebase``` folder. The FEN used in the example are arbitrary.
The [UCI Engine](https://github.com/Mufasaxi/NeuralBases/blob/main/src/tablebase/uci_prober_engine.py) is there in case of integration with a GUI or for setting up the prober to play with other engines (Note: this implementation works for endgames within the 6 man Syzygy Tablebases).

Under the ```neuralnet``` folder are all the necessary files for building, training, evaluating and using the neural network to act in place of the tablebase. The created sample data to train the neural network is created in the [train.py](https://github.com/Mufasaxi/NeuralBases/blob/main/src/neuralnet/train.py) file, and would take up around 3GBs of storage space. They require the presence of the aforementioned 6 man Syzygy Tablebase.
