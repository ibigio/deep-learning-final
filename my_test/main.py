import tensorflow as tf
import numpy as np

import operator as op
from functools import reduce

def nck(n, k):
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return numer // denom

class Model(tf.keras.Model):

  def __init__(self,max_in_num):
    """
    The model class inherits from tf.keras.Model.
    It stores the trainable weights as attributes.
    """
    super(Model, self).__init__()

    self.max_in_num = max_in_num
    self.max_out_num = nck(self.max_in_num,self.max_in_num // 2)

    self.hidden_size = 100
    self.batch_size = 1000

    self.dense_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
    self.dense_2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
    self.dense_3 = tf.keras.layers.Dense(self.max_out_num, activation='softmax')
    
  def call(self, inputs):
    """
    Forward pass, predicts labels given an input image using fully connected layers
    :return: the probabilites of each label
    """
    print(inputs)
    out_1 = self.dense_1(inputs)
    print(out_1)
    out_2 = self.dense_2(out_1)
    print(out_2)
    prbs = self.dense_3(out_2)
    print(prbs)

    return prbs
  
  def loss(self, predictions, labels):
    """
    Calculates the model loss
    :return: the loss of the model as a tensor
    """
    # Keras also has some nifty built in loss functions:
    losses = tf.keras.losses.categorical_crossentropy(predictions, labels)
    # print(predictions)
    # print(labels)
    # print(losses)
    return tf.reduce_sum(losses)  
  
  def accuracy(self, predictions, labels):
    """
    Calculates the model accuracy
    :return: the accuracy of the model as a tensor
    """

    correct_prediction = tf.equal(tf.argmax(predictions, 1),
                    tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(model, train_size):

    num_batches = max(train_size // model.batch_size, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(num_batches):

        # inputs, labels = generate_random_data(model.max_in_num, model.batch_size)
        inputs, labels = generate_complete_data(model.max_in_num)
        inputs = inputs / model.max_in_num
        labels = labels / model.max_in_num

        with tf.GradientTape() as tape:
            prbs = model(inputs)
            loss = model.loss(prbs, labels)
            accuracy = model.accuracy(prbs, labels)

            if i % 50 == 0:
                print(f"Accuracy on training set after {i} / {num_batches} training steps: {accuracy}")
        

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_size):

    inputs, labels = generate_complete_data(model.max_in_num)
    inputs = inputs / model.max_in_num
    labels = labels / model.max_in_num

    prbs = model(inputs)
    loss = model.loss(prbs, labels)
    accuracy = model.accuracy(prbs, labels)

    expected = tf.argmax(labels, 1)
    predicted = tf.argmax(prbs, 1)

    print('Exp:',expected)
    print('Pred:',predicted)

    correct_mask = tf.equal(expected, predicted)

    print(correct_mask)

    # print('=== Correct ===')
    # for i in range(len(inputs)):
    #     if correct_mask[i]:
    #         print("Input:", inputs[i])

    return loss, accuracy

def generate_random_data(max_in_num, amount):
    max_out_num = nck(max_in_num,max_in_num // 2)
    n_list = np.random.randint(1, max_in_num, amount)
    k_list = np.random.randint(n_list)

    inputs = tf.convert_to_tensor(list(zip(n_list, k_list)), dtype=tf.int32)
    labels = tf.convert_to_tensor([nck(n,k) for n,k in zip(n_list, k_list)], dtype=tf.int32)
    labels = tf.one_hot(labels,max_out_num)

    return inputs, labels

def generate_complete_data(max_in_num):
    max_out_num = nck(max_in_num,max_in_num // 2)
    n_list = range(1,max_in_num + 1)
    inputs = []
    for i in n_list:
        for j in range(1,i+1):
            inputs.append((i,j))
    labels = [nck(n,k) for n,k in inputs]

    inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels = tf.one_hot(labels,max_out_num)

    # print(len(inputs), len(labels))
    # print(inputs)
    # print(labels)

    return inputs, labels

def main():

    max_in_num = 10
    train_size = 10000000
    test_size = train_size // 10

    model = Model(max_in_num)

    train(model, train_size)
    loss, accuracy = test(model, test_size)
    print('Final Loss:', loss)
    print('Final Accuracy:', accuracy)

    
if __name__ == '__main__':
    main()

