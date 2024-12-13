from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib import pyplot


# load faces
data = load('facenet/faces_detection.npz')
X_test_faces = data['arr_2']
# load face embeddings
data = load('facenet/faces_embeddings.npz')
X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (X_train.shape[0], X_test.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(X_train)
testX = in_encoder.transform(X_test)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(Y_train)
Y_train = out_encoder.transform(Y_train)
Y_test = out_encoder.transform(Y_test)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
# predict
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)
# score
score_train = accuracy_score(Y_train, yhat_train)
score_test = accuracy_score(Y_test, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

for i in range(9):
# test model on a random example from the test dataset
    selection = choice([i for i in range(X_test.shape[0])])
    random_face_pixels = X_test_faces[selection]
    random_face_emb = X_test[selection]
    random_face_class = Y_test[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    # plot for fun
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()
