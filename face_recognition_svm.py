# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition 
from sklearn import svm
import os

encodings = []
names = []

#Model path 
model_save_path="trained_svn_model.clf"
# Training directory
train_dir = os.listdir('svm_examples/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("svm_examples/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("svm_examples/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains none or more than faces, print an error message and exit
        encodings.append(face)
        names.append(person)

#Create and train the SVC classifier
#Train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit([encodings,names])

#Saved Model
if model_save_path is not None:
        with open(model_save_path, 'ab') as f:
            pickle.dump(clf, f)

# Load the model
with open(model_save_path, 'rb') as f:
    svm_clf = pickle.load(f)
# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('test1.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)

print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = svm_clf.predict([test_image_enc])
    print(*name)
