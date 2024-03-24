import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from art.attacks.evasion import BasicIterativeMethod
from tensorflow.keras.layers import Conv2D, LSTM, SimpleRNN, GAN, Dense, RadialBasisFunction, Flatten, BatchNormalization, Reshape
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()
# Global variables
X_train, X_test, y_train, y_test = ''
dataset = 'WADI_attackdataLABLE.csv'

def split_dataset(X, y, test_size=0.2, random_state=42, folds=10):
    X = np.copy(X.values)
    y = np.copy(y.values)
    y[y == 1] = 0
    y[y == -1] = 1
    

    df_attack = X[np.where(y==1)[0]]
    df_normal = X[np.where(y==0)[0]]

    import random
    random.seed(random_state)
    X_normal_idx = random.sample(range(0, len(df_normal)), len(df_attack))
    X_normal_idx.sort()
    X_normal = df_normal[X_normal_idx]
    y_normal = y[np.where(y==0)[0]][X_normal_idx]
    X_attack = df_attack
    y_attack = y[np.where(y==1)[0]]

    X = np.vstack((X_normal,X_attack))
    y = np.hstack((y_normal,y_attack))

    train_size=1-test_size
    size = int(len((X_normal))/folds)
    X_train = np.vstack((X_normal[:size][:int(train_size*size)], X_attack[:size][:int(train_size*size)]))
    y_train = np.hstack((y_normal[:size][:int(train_size*size)], y_attack[:size][:int(train_size*size)]))
    X_test = np.vstack((X_normal[:size][int(train_size*size):], X_attack[:size][int(train_size*size):]))
    y_test = np.hstack((y_normal[:size][int(train_size*size):], y_attack[:size][int(train_size*size):]))
    for i in range(2,folds+1):
        aux = np.vstack((X_normal[size*(i-1):size*i][:int(train_size*size)], X_attack[size*(i-1):size*i][:int(train_size*size)]))
        X_train = np.vstack((X_train, aux))
        aux = np.hstack((y_normal[size*(i-1):size*i][:int(train_size*size)], y_attack[size*(i-1):size*i][:int(train_size*size)]))
        y_train = np.hstack((y_train, aux))

        aux = np.vstack((X_normal[size*(i-1):size*i][int(train_size*size):], X_attack[size*(i-1):size*i][int(train_size*size):]))
        X_test = np.vstack((X_test, aux))
        aux = np.hstack((y_normal[size*(i-1):size*i][int(train_size*size):], y_attack[size*(i-1):size*i][int(train_size*size):]))
        y_test = np.hstack((y_test, aux))
    return X_train, X_test, y_train, y_test

def process_data(dataset): 
    global X_train, X_test, y_train, y_test
    # Load the WADI dataset
    df = pd.read_csv(dataset)

    # Select the features you want to use, drop unnecessary columns
    df = df.drop(columns=['Row ','Date ','Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS'])

    # Drop the last two rows from the DataFrame
    # Use .iloc accessor to select the last two rows
    last_two_rows = df.iloc[-2:, :]
    df = df.drop(last_two_rows.index)

    # Checking if null values
    df.isnull().sum(axis=0)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(df.drop(columns=['Attack LABLE (1:No Attack, -1:Attack)']), df['Attack LABLE (1:No Attack, -1:Attack)'], test_size=0.2, random_state=42)

def get_scaler(type):
    if type == 'min_max':
        return MinMaxScaler()
    elif type == 'standard':
        return StandardScaler()
    elif type == 'robust':
        return RobustScaler()
    elif type == 'max_abs':
        return MaxAbsScaler()
    elif type == 'quantile':
        return QuantileTransformer(output_distribution='uniform')
    elif type == 'power_yeo_johnson':
        return PowerTransformer(method='yeo-johnson')
    elif type == 'power_box_cox':
        return PowerTransformer(method='box-cox')
    else:
        raise ValueError("Invalid scaler type. Please choose a valid type.")

def use_scaler(type_scaler):
    scaler = get_scaler(type_scaler)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

def create_neural_network(type):
    model = Sequential()
    if type == 1:  # Convolutional Neural Networks (CNNs)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))

    elif type == 2:  # Long Short Term Memory Networks (LSTMs)
        model.add(LSTM(50, input_shape=(timesteps, features)))
        model.add(Dense(1, activation='sigmoid'))

    elif type == 3:  # Recurrent Neural Networks (RNNs)
        model.add(SimpleRNN(50, input_shape=(timesteps, features)))
        model.add(Dense(1, activation='sigmoid'))

    elif type == 4:  # Generative Adversarial Networks (GANs)
        # Implement GAN model architecture here
        pass

    elif type == 5:  # Radial Basis Function Networks (RBFNs)
        model.add(RadialBasisFunction(10, input_shape=(X_train.shape[1],)))

    elif type == 6:  # Multilayer Perceptrons (MLPs)
        model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

    elif type == 7:  # Self Organizing Maps (SOMs)
        # Implement SOM model architecture here
        pass

    elif type == 8:  # Deep Belief Networks (DBNs)
        # Implement DBN model architecture here
        pass

    elif type == 9:  # Restricted Boltzmann Machines( RBMs)
        # Implement RBM model architecture here
        pass

    elif type == 10:  # Autoencoders
        model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))

    else:
        raise ValueError("Invalid neural network type. Please choose a type from 1 to 10.")

    return model


def create_fit_save_model(type_model):
    # Define the model architecture
    model = create_neural_network(type_model)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create a KerasClassifier instance
    classifier = KerasClassifier(model=model, clip_values=(0, 1))

    y_train_categorical = to_categorical(y_train, num_classes=2)

    # Train the model
    classifier.fit(X_train, y_train_categorical, batch_size=32)

    # Save the trained model
    model.save('modelMinMax4L.h5')


def evaluate_model(model):
    # Evaluate the load model
    y_pred = model.predict(X_test)
    y_pred_adv = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_test, y_pred_adv, average='macro')
    recall = recall_score(y_test, y_pred_adv, average='macro')
    precision = precision_score(y_test, y_pred_adv, average='macro')
    print('F1 score:', f1)
    print('Recall score:', recall)
    print('Precision score:', precision)

if __name__ == '__main__':
    process_data(dataset)
    use_scaler('min_max')
    model = create_fit_save_model(6)
    evaluate_model(model)