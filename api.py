from flask import Flask, request
import numpy as np


from keras.models import model_from_json
import cv2 as cv
app = Flask(__name__)
#global vars for easy reusability
global model




def init():
	json_file = open('model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model/model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	return loaded_model

model = init()

@app.route('/predict/',methods=['POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image

	imgData = request.get_data()
	nparr = np.fromstring(imgData, np.uint8)
	# decode image
	x = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
	x = x.T #transposing to match EMINST's format
	x = cv.resize(x, (28, 28))
	x = np.invert(x)
	x = x.astype('float32')
	x /= 255
	x = x.reshape(-1, 28, 28, 1)

	#perform the prediction
	out = model.predict(x)
	#print(out)
	#print(np.argmax(out, axis=1))
	#convert the response to a string

	from string import ascii_uppercase

	out = np.argmax(out, axis=1)[0]
	out = ascii_uppercase[out]
	response = out
	return response

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8015)

