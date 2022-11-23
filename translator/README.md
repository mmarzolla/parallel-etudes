# Automatic translator

This folder contains the automatic python translator that uses Google API
to translate the text to the chosen language.
Currently this traslator has only been tested with dummy scenarios and has
yet to be tested with a real application (for example it has only been
tested with CUDA files).

## Comments
The comments to the script are going to be added soon to better understand
how the translator function.
The comments are still missing due to last minute changes and the will to
give a better description of the thought process that i took writing this
script and to better understand each step of the process.

## How to run
First of all you will need to install the required libraries used within the
script.
By using the following command pip will automatically install every library
that is withing the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

To run the traslator script:

```bash
python translator.py
```