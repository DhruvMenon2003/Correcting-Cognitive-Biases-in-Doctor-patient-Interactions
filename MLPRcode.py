import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to load and preprocess the data
def load_and_preprocess_data(filepath):

    df = pd.read_excel(filepath)
    
    # Drop rows with any null values in 'section_text' or the label column
    df.dropna(subset=['section_text', 'Bias Present? If yes, which type?'], inplace=True)
    
    # Remove any 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df

# Function to prepare features and labels
def prepare_features_labels(df):
   
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['section_text']).toarray()
    
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Bias Present? If yes, which type?'])
    
    return X, y, label_encoder, vectorizer

# Function to build the neural network model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(filepath):
    
    df = load_and_preprocess_data(filepath)
    
   
    X, y, label_encoder, vectorizer = prepare_features_labels(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Build the model
    model = build_model(X_train.shape[1], len(np.unique(y_train)))
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        class_weight=class_weights_dict,
                        callbacks=[early_stopping],
                        verbose=1)
    
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test_categorical)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    # Predict on the test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix Display
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
   
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(y)), test_size=0.2, random_state=42)
    
    # Display sample text with predicted and actual labels using indices_test to fetch the actual texts
    sample_texts = df['section_text'].iloc[indices_test]
    for i, (text, pred, true) in enumerate(zip(sample_texts[:10], y_pred[:10], y_test[:10])):
        print(f"Sample {i+1}:")
        print(f"Text: {text[:60]}...")
        print(f"Predicted Label: {label_encoder.inverse_transform([pred])[0]}")
        print(f"Actual Label: {label_encoder.inverse_transform([true])[0]}")
        print("")


if __name__ == '__main__':
    main(r"C:\Users\lenovo\Downloads\cognitive_biases_dataset_fully_modified.xlsx") 
