#import numpy as np # Importando Numpy porque dependemos de ella.
import math
import numpy as np
from time import sleep


#Todas las funciones están pensadas para ser implementadas sobre arrays de Numpy
def identity(x):
    return x

def identityderiv(x):
    return np.ones(x.shape)

def relu(x):
    return (x>=0)*x         #Problemas con esto

def reluderiv(x):
    return (x>=0)*1        #Problemas con esto

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidderiv(x):
    s = sigmoid(x)       
    s_1 = 1-s     
    return s*s_1
### Por el momento esta red neuronal usará solamente la MSE como función de costos, 
#   en el futuro se considerará ampliarla para que pueda ser entrenada usando otras funciones de costo.

## CLASE PARA CREAR CAPAS 
class layer():      #¿Hará falta una capa para nodos?
    def __init__(self,tipo,datos,activacion):
        self.tipo = tipo
        self.activacion = activacion
        self.input = datos
        activ_expr = activacion+'('+'datos'+')'
        self.valor = eval(activ_expr)
    def deriv(self):     
        activprim_expr = self.activacion+'deriv('+'self.input'+')'
        self.derivada = eval(activprim_expr)
        return self.derivada
    def dim(self):
        return self.valor.shape
    def __str__(self):
        return str(self.valor)
# Pruebas para ver que funcione la clase
prueba = np.array([1,5,-48,-7,8,9,-55,0,0.0001,454,-888])
prueba.shape = (1,11)
# layer_prueba = layer('none',prueba,'identity')
# print(layer_prueba)
# print(layer_prueba.prim())

class NN():                         
    def __init__(self,size):
        self.info = [['input',size,'identity']] 
        self.WEIGHTS = []   #Atributo de los pesos
        self.LAYERS = [layer('input',np.random.random((1,size)),self.info[0][2])]
        self.DELTAS = []    
    
    def add_layer(self,type,size,activation):   #Método para agregar capas, con cada capa se agrega una nueva matriz de pesos y un delta
        c = len(self.LAYERS)
        self.LAYERS.append(layer(type, np.random.random((1,size)),activation))
        self.WEIGHTS.append(0.2*np.random.random((self.LAYERS[c-1].dim()[1], size))-0.1) #Interesante forma de inizializar los pesos.
        self.DELTAS.append(layer(type, np.random.random((1,size)),activation))
    
    def predict(self,X):    #Método para usar el modelo después del entrenamiento
        prediction = X.copy() 
        for j in range(len(self.WEIGHTS)):
            a = prediction.dot(self.WEIGHTS[j])
            L = layer(self.LAYERS[j+1].tipo,a,self.LAYERS[j+1].activacion)  
            prediction = L.valor
        return prediction
    
    def train(self,X,Y,alpha,iterations):  #Rutina de entrenamiento
        data_size = len(X)
        for j in range(iterations):
            error,preds = (0.0,0)
            for i in range(data_size):
                
                ####FORWARDPASS
                self.LAYERS[0] = layer('input',X[i:i+1],'identity')
                for k in range(len(self.WEIGHTS)):
                    z_values = self.LAYERS[k].valor.dot(self.WEIGHTS[k])
                    a_values = layer(self.LAYERS[k+1].tipo,z_values,self.LAYERS[k+1].activacion)
                    self.LAYERS[k+1] = a_values
                #END FORWARDPASS
                
                #Algunas métricas
                error += np.sum((Y[i:i+1]-self.LAYERS[len(self.WEIGHTS)].valor)**2)
                preds += int(np.argmax(self.LAYERS[len(self.WEIGHTS)].valor) == np.argmax(Y[i:i+1]))
                #
                
                ## Actualizando deltas
                self.DELTAS[len(self.DELTAS)-1] = (Y[i:i+1]-self.LAYERS[len(self.DELTAS)].valor
                )*self.LAYERS[len(self.DELTAS)].deriv()
                for k in range(len(self.DELTAS)-2,-1,-1):
                    self.DELTAS[k] = self.DELTAS[k+1].dot(self.WEIGHTS[k+1].T)*self.LAYERS[k+1].deriv()
                #Fin de actualización de deltas
                
                #### BACKWARDPASS
                for k in range(len(self.WEIGHTS)-1,-1,-1):
                    self.WEIGHTS[k] += alpha * self.LAYERS[k].valor.T.dot(self.DELTAS[k])
            #Mostrando el progreso del entrenamiento        
            print("\rI",f" :{j+1}/{iterations} "," Error asociado: ",error/data_size,
            f" Ejemplos clasificadas correctamente:{preds}/{data_size}",sep='',end='',flush=True)
            
                






np.random.seed(1)

from keras.datasets import mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()
images, labels  = (x_train[0:1000].reshape(1000,28**2)/255,y_train[0:1000])
one_hot_labels = np.zeros((len(labels),10))

for i,j in enumerate(labels):
    one_hot_labels[i][j] = 1
labels = one_hot_labels
##Test images from MNIST dataset for model validation and hyperparameter tunning
# test_images = x_test.reshape(len(x_test),28**2)/255
# test_labels = np.zeros((len(y_test),10))
# for i,j in enumerate(y_test):
    # test_labels[i][j] = 1

lr,iterations,hidden_size,pixels_per_image,num_labels = (0.005,350,40,784,10)   #Hyperparameters for fun


nn_mnist = NN(784)
nn_mnist.add_layer('hidden', size=40, activation='relu')
nn_mnist.add_layer('hidden', size=60, activation='sigmoid')
# nn_mnist.add_layer('output', size=10, activation='sigmoid')
nn_mnist.add_layer('output', size=10, activation='identity')
nn_mnist.train(images, labels, lr, iterations)
