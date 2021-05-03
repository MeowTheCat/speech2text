import os
from tensorflow import keras

keras.utils.get_file(
    os.path.join("/Users/miaowu/Documents/GitHub/speech2text/data", "data.tar.gz"),
    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    extract=True,
    archive_format="tar",
    cache_dir=".",
)
