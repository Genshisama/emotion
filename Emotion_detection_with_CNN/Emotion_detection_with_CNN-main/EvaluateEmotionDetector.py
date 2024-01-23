
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, ConfusionMatrixDisplay


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# do prediction on test data
predictions = emotion_model.predict_generator(test_generator)

# see predictions
for result in predictions:
        max_index = int(np.argmax(result))
        print(emotion_dict[max_index])

print("-----------------------------------------------------------------")
# confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))

print("\nClassification Report:")
print(classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=list(emotion_dict.values())))

# Precision, Recall, F1-score
precision, recall, fscore, _ = precision_recall_fscore_support(test_generator.classes, predictions.argmax(axis=1))

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(emotion_dict)), precision, marker='o', label='Precision')
plt.plot(range(len(emotion_dict)), recall, marker='o', label='Recall')
plt.plot(range(len(emotion_dict)), fscore, marker='o', label='F1-score')
plt.xticks(range(len(emotion_dict)), list(emotion_dict.values()), rotation=45)
plt.title('Precision, Recall, and F1-score for Each Emotion Class')
plt.xlabel('Emotion Class')
plt.ylabel('Score')
plt.legend()
plt.show()