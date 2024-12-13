from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras_facenet import FaceNet

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# load the face dataset
data = load('facenet/faces_detection.npz')
X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# load the facenet model
base_model = FaceNet()
base_model = base_model.model

x = base_model.output
x = Dense(128, activation='relu')(x)
X = Dense(128, activation='softmax')(x)
x = Dropout(0.5)(x)

model = Model(inputs = base_model.input, outputs = x)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
print('Model Loaded')
# convert each face in the train set to an embedding
X_train_new = list()
for face_pixels in X_train:
	embedding = get_embedding(model, face_pixels)
	X_train_new.append(embedding)
X_train_new = asarray(X_train_new)
print(X_train_new.shape)
# convert each face in the test set to an embedding
X_test_new = list()
for face_pixels in X_test:
	embedding = get_embedding(model, face_pixels)
	X_test_new.append(embedding)
X_test_new = asarray(X_test_new)
print(X_test_new.shape)
# save arrays to one file in compressed format
savez_compressed('facenet/faces_embeddings.npz', X_train_new, Y_train, X_test_new, Y_test)