import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import numpy as np
import time
import tensorboard
folder = "6000"
num_of_data =6000
def extract_training_data_inputs():

    temperature = []
    catalyst_conc= []
    cocatalyst_conc = []

    files = os.listdir("C:\\Users\\Kevin\\Desktop\\NN_data\\"+folder+"\\parsfiles")#PycharmProjects\\NNfun\\pars"
    par_files = [file for file in files if file.startswith("par_")] #vypada to jako necitelny kod, ale to prvni file je
    #expression takze bych tam treba mohl dat float(files) a kazdou tu file to iterativne prevede na float verzi

    par_files = par_files[0:num_of_data]

    for par_file in par_files:
            with open("C:\\Users\\Kevin\\Desktop\\NN_data\\"+folder+"\\parsfiles\\"+par_file, "r") as f:
                lines = f.readlines()
                temperature.append(float(lines[0]))
                catalyst_conc.append(float(lines[1]))
                cocatalyst_conc.append(float(lines[2]))

    return temperature,catalyst_conc,cocatalyst_conc

def extract_training_data_outputs():
    values = []
    Mn_ODE = []
    Mn_MC = []
    Mw_ODE = []
    Mw_MC = []
    G_crossover = []
    G_plateau = []
    files = os.listdir("C:\\Users\\Kevin\\Desktop\\NN_data\\"+folder+"\\outfiles")#PycharmProjects\\NNfun\\pars"
    out_files = [file for file in files if file.startswith("out_")]  # vypada to jako necitelny kod, ale to prvni file
    #je expression takze bych tam treba mohl dat float(files) a kazdou tu file to iterativne prevede na float verzi
    out_files = out_files[0:num_of_data]
    for out_file in out_files:
        with open("C:\\Users\\Kevin\\Desktop\\NN_data\\"+folder+"\\outfiles\\"+out_file, "r") as f:
            for line in f:
                if "(" in line:  # finds the correct lines
                    values.append(line)

            Mn_ODE.append(float(values[0][20:41]))
            Mn_MC.append(float(values[1][20:41]))
            Mw_ODE.append(float(values[2][20:41]))
            Mw_MC.append(float(values[3][20:41]))
            G_crossover.append(float(values[4][20:41]))
            G_plateau.append(float(values[5][20:41]))
            values = []
    return Mn_ODE,Mw_ODE,G_crossover,G_plateau


def modify_training_data(data_input,data_output):

    temp_data = data_input[0]
    cat_conc_data = data_input[1]
    cocat_conc_data = data_input[2]
    mn_ode_data = data_output[0]
    mw_ode_data = data_output[1]

    G_crossover_data = [x/20000 for x in data_output[2]] #/20000
    G_plateau_data = [x/1e5 for x in data_output[3]]     #/1e5

    dataset = pd.DataFrame(list(zip(mn_ode_data, mw_ode_data,G_crossover_data,G_plateau_data,temp_data,cat_conc_data,cocat_conc_data)),
            columns =["Mn_ODE", "Mw_ODE","G_crossover","G_plateau","temperature","catalyst_conc","cocatalyst_conc"])
    print(dataset.tail().to_string())


    # sns.pairplot(dataset[["Mn_ODE", "Mw_ODE","G_crossover","G_plateau","temperature","catalyst_conc","cocatalyst_conc"]], diag_kind='kde')
    # plt.show()
    #vybiram si jaka cast dataset pujde na trenovani site a jak na testovani (frac), random state je asi
    #kdybych chtel jakoby stejny seed toho jake indexy se rozdeli do jakych datasetu

    train_dataset = dataset.sample(frac=0.9, random_state=3)
    test_dataset = dataset.drop(train_dataset.index)

    #nejake statistiky kdyby me to zajimalo
    train_stats = train_dataset.describe()
    train_stats.pop("Mn_ODE")
    train_stats.pop('Mw_ODE')
    train_stats.pop("G_crossover")
    train_stats.pop("G_plateau")
    train_stats = train_stats.transpose()
    print(train_stats.to_string())
    #spravne hodnoty pro ANN na trenovani
    train_labels1 = train_dataset.pop("Mn_ODE")
    train_labels2 = train_dataset.pop('Mw_ODE')
    train_labels3 = train_dataset.pop("G_crossover")
    train_labels4 = train_dataset.pop("G_plateau")
    train_labels = pd.concat([train_labels1, train_labels2.reindex(train_labels1.index),
                        train_labels3.reindex(train_labels1.index),train_labels4.reindex(train_labels1.index)], axis=1)

    # spravne hodnoty pro ANN na testovani
    test_labels1 = test_dataset.pop("Mn_ODE")
    test_labels2 = test_dataset.pop('Mw_ODE')
    test_labels3 = test_dataset.pop("G_crossover")
    test_labels4 = test_dataset.pop("G_plateau")
    test_labels = pd.concat([test_labels1, test_labels2.reindex(test_labels1.index),
                            test_labels3.reindex(test_labels1.index),test_labels4.reindex(test_labels1.index)], axis=1)
    return train_dataset,test_dataset,train_stats,train_labels,test_labels
def norm(x,train_stats):
    """Standard normalization of values between 0 and 1, in the comment is so called
    Z-score normalization, the mean of the normalized values is 0
    and the standard deviation of the normalized values is 1"""
    #(x-train_stats["mean"])/train_stats["std"]
    return (x-x.min())/(x.max()-x.min())


def build_model(train_dataset):
    # Create a Sequential model, which is a linear stack of layers, simple feed-forward NN
    model = keras.Sequential([
        # Add a dense (fully connected) layer with 64 units and ReLU activation function.
        layers.Dense(300, activation=tf.nn.tanh, input_shape=[len(train_dataset.keys())]),  #input shape is 3 (Tdeg,cat_conc,cocat_conc)
        #layers.Dropout(0),
        layers.Dense(100, activation=tf.nn.tanh),



        # Add another dense layer with 64 units and ReLU activation function.














        # Add a final dense layer with 2 units (common for regression tasks). There is no activation (the activation is linear, y=x,
        #whatever the input to the node is, that is the final values i guess
        layers.Dense(4)
    ])

    # Define an optimizer (RMSprop, root mean square propagation) with a learning rate of 0.001
    # RMSprop uses a moving average of the squared gradients to scale the learning rates, parameters that have large
    # gradients will have smaller learning rates and vice versa (With gradient descent the LR is fixed for all parameters)


    # Compile the model with loss function, optimizer, and evaluation metrics
    model.compile(
        loss="mape",  # Mean Squared Error
        optimizer=tf.keras.optimizers.RMSprop(), # 0.001 is learning rate
        metrics=["mae", "mse","mape"]  # Metrics to track during training (Mean Absolute Error and Mean Squared Error)
    )
    return model

def plot_history(history):

    hist = pd.DataFrame(history.history)
    print(hist)
    hist["epoch"] = history.epoch
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error")
    plt.plot(hist["epoch"],hist["mape"],label = "Train error")
    plt.plot(hist["epoch"],hist["val_mape"],label = "Validation error")
    plt.legend()
    plt.ylim([0,6])
    plt.show()

def main_function():
    data_input = extract_training_data_inputs()
    data_output = extract_training_data_outputs()
    modified_data = modify_training_data(data_input,data_output)

    train_dataset = modified_data[0]
    test_dataset = modified_data[1]
    train_stats = modified_data[2]
    train_labels = modified_data[3]
    test_labels = modified_data[4]
    print(train_dataset.tail())
    print("mezera")
    print(train_labels.tail())
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)
    # The 'build_model' function creates and configures a neural network model for a regression task.
    model = build_model(train_dataset)

    EPOCHS = 2400


    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300)
    history = model.fit(
        normed_train_data,train_labels,epochs=EPOCHS,validation_split=0.2,verbose=2,callbacks=early_stop)
    loss, mae, mse, mape = model.evaluate(normed_test_data, test_labels, verbose=0)
    print("\nTesting set Mean Abs Percentage Error:" + str(round(mape, 3)))
    plot_history(history)

    test_predictions = model.predict(normed_test_data)  #add.flaten() if something does not work
    print(test_predictions)
    start = time.time()
    #TODO musis to tam dat normovane, pak uz to pujde
    print(model(np.array([[0.7,0.8,0.7]])))
    end = time.time()
    print(f"Time to evaluate one set of reaction conditions {(end - start)*1000} miliseconds")

    num_of_test_points =40

    plt.figure(figsize=(4, 4))
    #plt.title("SM vs deductive model, epochs = "+str(EPOCHS)+", MAPE = " +str(round(mape, 2))+" % \n ",fontsize=15)
    plt.scatter(test_labels["Mw_ODE"][0:num_of_test_points], test_predictions[:, 1][0:num_of_test_points],
                label="$M_w$", facecolors='none', edgecolors='r')
    plt.scatter(test_labels["Mn_ODE"][0:num_of_test_points], test_predictions[:, 0][0:num_of_test_points],
                label="$M_n$", facecolors='none', edgecolors='b')
    #plt.xlabel("$M_w$ [kg/mol] and $M_n$ [kg/mol] predicted by deductive model", fontsize=13,labelpad=10)
    #plt.ylabel("$M_w$ [kg/mol] and $M_n$ [kg/mol] predicted by surrogate model",fontsize=13,labelpad=10)

    x= range(0,110)
    y=x
    plt.plot(x,y,color="black",linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=13)
    plt.savefig('Mn.png', dpi=700)
    plt.show()


    plt.figure(figsize=(4, 4))
    #plt.title("SM vs deductive model, epochs = " + str(EPOCHS) + ", MAPE = " + str(round(mape, 2)) + " % \n ",
    #          fontsize=15)
    plt.scatter((test_labels["G_crossover"]*20000)[0:num_of_test_points], (test_predictions[:, 2]*20000)[0:num_of_test_points], label="$G_c$", facecolors='none', edgecolors='g')
    plt.scatter((test_labels["G_plateau"]*1e5)[0:num_of_test_points], (test_predictions[:, 3]*1e5)[0:num_of_test_points], label="$G^0_N$", facecolors='none', edgecolors='y')
    #plt.xlabel("$G_c$ [Pa] and $G^0_N$ [Pa] predicted by deductive model", fontsize=13, labelpad=10)
    #plt.ylabel("$G_c$ [Pa] and $G_N^0$ [Pa] predicted by surrogate model", fontsize=13, labelpad=10)
    x=[0,10000000]
    y = x
    plt.plot(x, y, color="black", linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=13)
    plt.xscale(value="log")
    plt.yscale(value="log")
    plt.savefig('G_mod.png', dpi=700)
    plt.show()

main_function()