import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import numpy as np
import ode_PLA
import time
import tensorboard
sampler = qmc.LatinHypercube(d=3,seed=2)
sample = sampler.random(n=800)
# [Tdeg,catalyst concentration ,cocatalyst concentration]
l_bounds = [140,0.5,40]
u_bounds = [160,2,120]


training_data = qmc.scale(sample,l_bounds,u_bounds)
Tdeg_data = training_data[:,0]
cat_conc_data = training_data[:,1]
cocat_conc_data = training_data[:,2]
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# label_pad = 10
# font_S =14
# ax.scatter(training_data[:,0],training_data[:,1],training_data[:,2],s=50)
# ax.set_xlabel('Reaction temperature [°C]',fontsize=font_S,labelpad=label_pad)
# ax.set_ylabel('Concentration of catalyst [mol/m3]',fontsize=font_S,labelpad=label_pad)
# ax.set_zlabel('Concentration of co-catalyst [mol/m3]',fontsize=font_S,labelpad=label_pad)
# ax.tick_params(labelsize=font_S)
# plt.show()
pocetni_list = []
hmotnostni_list = []
konverze_list = []
for i in range(0,len(Tdeg_data)):
    outputs = ode_PLA.main_func(Tdeg_data[i],cat_conc_data[i],cocat_conc_data[i])
    pocetni = outputs[0]
    hmotnostni = outputs[1]
    konverze = outputs[2]
    pocetni_list.append(pocetni)
    hmotnostni_list.append(hmotnostni)
    konverze_list.append(konverze)

dataset = pd.DataFrame(list(zip(pocetni_list, hmotnostni_list,konverze_list,Tdeg_data,cat_conc_data,cocat_conc_data)),
               columns =["Pocetni", "hmotnostni","konverze","teplota","kat konc","kokat konc"])
print(dataset.tail())
# plt.scatter(pocetni_list,Tdeg_data)
#plt.show()
train_dataset = dataset.sample(frac=0.998, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#sns.pairplot(train_dataset[['Pocetni', 'hmotnostni', 'teplota', 'kat konc', "kokat konc"]], diag_kind='kde')
#plt.show()

# Remove the 'hmotnostni' and "pocetni" columns from the statistical summary since it's the target variable
train_stats = train_dataset.describe()
# Remove the 'hmotnostni' and "pocetni" columns from the statistical summary since it's the target variable
train_stats.pop("Pocetni")
train_stats.pop("hmotnostni")
train_stats.pop("konverze")
train_stats = train_stats.transpose()
print(train_stats)

train_labels1 = train_dataset.pop('Pocetni')
train_labels2 = train_dataset.pop('hmotnostni')
train_labels3 = train_dataset.pop("konverze")
train_labels = pd.concat([train_labels1, train_labels2.reindex(train_labels1.index),train_labels3.reindex(train_labels1.index)], axis=1)
print(train_labels)


test_labels1 = test_dataset.pop('Pocetni')
test_labels2 = test_dataset.pop('hmotnostni')
test_labels3 = test_dataset.pop("konverze")
test_labels = pd.concat([test_labels1, test_labels2.reindex(test_labels1.index),test_labels3.reindex(test_labels1.index)], axis=1)
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
        layers.Dense(16, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),  #input shape is 3 (Tdeg,cat_conc,cocat_conc)

        # Add another dense layer with 64 units and ReLU activation function.
        layers.Dense(12, activation=tf.nn.relu),




        # Add a final dense layer with 2 units (common for regression tasks). There is no activation (the activation is linear, y=x,
        #whatever the input to the node is, that is the final values i guess
        layers.Dense(3)
    ])

    # Define an optimizer (RMSprop, root mean square propagation) with a learning rate of 0.001
    # RMSprop uses a moving average of the squared gradients to scale the learning rates, parameters that have large
    # gradients will have smaller learning rates and vice versa (With gradient descent the LR is fixed for all parameters)


    # Compile the model with loss function, optimizer, and evaluation metrics
    model.compile(
        loss="mse",  # Mean Squared Error
        optimizer=tf.keras.optimizers.RMSprop(0.001), # 0.001 is learning rate
        metrics=["mae", "mse","mape"]  # Metrics to track during training (Mean Absolute Error and Mean Squared Error)
    )


    return model
# The 'build_model' function creates and configures a neural network model for a regression task.
model = build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 ==0:
            print("")
        print(".",end="")
# Define the Keras TensorBoard callback.
#tb_callback = keras.callbacks.TensorBoard(log_dir="./logs")
EPOCHS = 400
# patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",patience=20)
history = model.fit(
    normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[PrintDot()] )
# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch


def plot_history(history):

    hist = pd.DataFrame(history.history)
    print(hist)
    hist["epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"],hist["mae"],label = "Train error")
    plt.plot(hist["epoch"],hist["val_mae"],label = "Validation error")
    plt.legend()
    plt.ylim([0,2])
    plt.show()
plot_history(history)
loss,mae,mse,mape = model.evaluate(normed_test_data,test_labels,verbose=0)
print("\nTesting set Mean Abs Percentage Error:"+ str(round(mape, 3)))

test_predictions = model.predict(normed_test_data)  #add.flaten() if something does not work
T = (160-150)/5.7696
kat = (1-1.242957)/0.437848
kokat = (60-79.9586)/23.134

start = time.time()
print(model(np.array([[T,kat,kokat]])))
end = time.time()
print(end - start)



plt.title("SM vs original model, epochs = "+str(EPOCHS)+", MAPE = " +str(round(mape, 2))+" % \n ",fontsize=15)
plt.scatter(test_labels["hmotnostni"],test_predictions[:,1], label = "$M_w$",facecolors='none', edgecolors='r')
plt.scatter(test_labels["Pocetni"],test_predictions[:,0], label = "$M_n$",facecolors='none', edgecolors='b')
plt.scatter(test_labels["konverze"],test_predictions[:,2], label = "$X$",facecolors='none', edgecolors='g')
plt.xlabel("True values of $M_w$,$M_n$ [kg/mol] and $X$ [%]", fontsize=13)
plt.ylabel("Predicted values of  $M_w$,$M_n$ [kg/mol] and $X$ [%]",fontsize=13)
x= range(0,102)
y=x
plt.plot(x,y,color="black",linewidth=0.5)
plt.tick_params(labelsize=13)
plt.legend(fontsize=13)
plt.show()
#%tensorboard --logdir=logs/
# PDI = test_predictions[:,1]/test_predictions[:,0]

"""Problem s casem, kdyz to je online, bude potreba to delat pro ruzne casy ne? ze parametr bude i cas? = recurrent networks

Model se dokaze naucit poměrně přesně se sítí 64*64*2, MAE (mean absolute error = 0.5), trénován na 100 000 hodnotách.
Hammersley

"""

