
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow import keras
from keras import layers
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Read the dataset from the specified URL, using the defined column names.
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# Create a copy of the 'raw_dataset' and store it in a new DataFrame called 'dataset'.
dataset = raw_dataset.copy()

# Display the last 5 rows of the 'dataset' for inspection.
print(dataset.tail())
# find unknown values
print(dataset.isna().sum())
# get rid of unknown values
dataset = dataset.dropna()

origin = dataset.pop("Origin")
dataset["USA"] = (origin==1)*1.0
dataset["Europe"] = (origin==2)*1.0
dataset["Japan"] = (origin==3)*1.0

# the output look like the origin columns have booleans, but I think it is just a different way of showing 1 and 0
print(dataset.tail())
print("above is the dataset tail")
# 80% of data is in training dataset
train_dataset = dataset.sample(frac=0.8, random_state=0)
print(train_dataset.tail())
# the rest is in the test_dataset
test_dataset = dataset.drop(train_dataset.index)
# The top row suggests that the fuel efficiency (MPG) is a function of all the other parameters.
# The other rows indicate they are functions of each other. KDE plots on the diagonal, spojité histogramy
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()
# Generate statistical summary for the training dataset
train_stats = train_dataset.describe()
# Remove the 'MPG' column from the statistical summary since it's the target variable
train_stats.pop("MPG")

train_stats = train_stats.transpose()
print(train_stats)
# Remove the 'MPG' column from the training dataset and store it in train_labels
train_labels = train_dataset.pop('MPG')
# Remove the 'MPG' column from the test dataset and store it in test_labels
test_labels = test_dataset.pop('MPG')

# We need to normalize the data, so that all of it is in the same order of magnitude
def norm(x):
    """Z-score normalization, the mean of the normalized values is 0
    and the standard deviation of the normalized values is 1"""

    return (x-train_stats["mean"])/train_stats["std"]
normed_train_data = norm(train_dataset)
normed_test_data = norm (test_dataset)
def build_model():
    # Create a Sequential model, which is a linear stack of layers, simple feed-forward NN
    model = keras.Sequential([
        # Add a dense (fully connected) layer with 64 units and ReLU activation function.
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),

        # Add another dense layer with 64 units and ReLU activation function.
        layers.Dense(64, activation=tf.nn.relu),

        # Add a final dense layer with 1 unit (common for regression tasks, as it outputs a single value).
        layers.Dense(1)
    ])

    # Define an optimizer (RMSprop, root mean square propagation) with a learning rate of 0.001
    # RMSprop uses a moving average of the squared gradients to scale the learning rates, parameters that have large
    # gradients will have smaller learning rates and vice versa (With gradient descent the LR is fixed for all parameters)
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # Compile the model with loss function, optimizer, and evaluation metrics
    model.compile(
        loss="mse",  # Mean Squared Error
        optimizer=optimizer,
        metrics=["mae", "mse"]  # Metrics to track during training (Mean Absolute Error and Mean Squared Error)
    )


    return model
# The 'build_model' function creates and configures a neural network model for a regression task.
model = build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 ==0:print("")
        print(".",end="")

EPOCHS = 1000
# patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)
history = model.fit(
    normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[early_stop,PrintDot()])
# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.plot(hist["epoch"],hist["mae"],label = "Train error")
    plt.plot(hist["epoch"],hist["val_mae"],label = "Validation error")
    plt.legend()
    plt.ylim([0,5])
    plt.show()
plot_history(history)
loss,mae,mse = model.evaluate(normed_test_data,test_labels,verbose=0)
print("\nTesting set Mean Abs Error:"+ str(round(mae, 3))+" MPG")

test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels,test_predictions)
plt.xlabel("True values of MPG")
plt.ylabel("Predicted values of MPG")
x= range(0,50)
print(x)
y=x
plt.plot(x,y)
plt.show()
