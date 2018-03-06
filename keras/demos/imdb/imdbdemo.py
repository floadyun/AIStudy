from keras import models
from keras.preprocessing import text
import numpy as np

model = models.load_model('imdb.h5')

test = 'very good'

textArray = text.text_to_word_sequence(test)

model.predict(textArray)
