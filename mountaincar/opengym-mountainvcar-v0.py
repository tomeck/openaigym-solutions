# AI Gym MountainCar-v0
# JTE 03/09/18
# Solved with 100% accuracy (100/100) using position update
# as the state change that reinforces a position action
#
# See https://github.com/openai/gym/wiki/MountainCar-v0

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median 
from collections import Counter

# learning rate
LR = 1e-3
goal_steps = 200
training_games = 10000

# Here's the AI gym environment we're using
env = gym.make('MountainCar-v0')

# TODO read from env.observation_space.low/high
V_DELTA_MIN = 0.001
P_DELTA_MIN = 0.00
V_MIN = 0 # -0.07
V_MAX = 0.07
P_MIN = 0 # -1.2
P_MAX = 0.6
TRAIN_DATA_FILENAME = 'good-pos-train.npy'
TRAINED_MODEL_FILENAME = 'pos-trained.tflearn'

print('Environment space: ', env.observation_space)
print('Environment space low: ', env.observation_space.low)
print('Environment space high: ', env.observation_space.high)
print('Action space: ', env.action_space)
print('Num actions: ', env.action_space.n)


# Not for real use: just shows what happens if you
# play the game by randomly sampling from the 
# game's action space
def some_random_games_first():
    for _ in range(5):
        env.reset()
        for _ in range(goal_steps):
            env.render()
            action = env.action_space.sample()  # get a random action
            observation, reward, done, info = env.step(action) # execute the action
            if done:
                break

#some_random_games_first()

def update_training_set_velocity(new_state, prev_state, action, training_data):
    
    # Keep this move if it increased the velocity
    v2 = new_state[1]+V_MIN
    v1 = prev_state[1]+V_MIN
    if( (v2 - v1) > V_DELTA_MIN):
        print('Velocity increased ', (v2-v1))
        # Encode action as 1-hot vector
        if action == 1:  
            one_hot = [0,1,0]
        elif action == 0:
            one_hot = [0,0,1]
        else:
            one_hot = [1,0,0]

        # Remember this move - it's a good one!
        training_data.append([prev_state, one_hot])

def update_training_set_position(new_state, prev_state, action, training_data):
    
    # Keep this move if it increased the position
    p2 = new_state[0]+P_MIN
    p1 = prev_state[0]+P_MIN
    if( p2 > p1):
        print('Position increased ', (p2-p1))
        # Encode action as 1-hot vector
        if action == 1:  
            one_hot = [0,1,0]
        elif action == 0:
            one_hot = [0,0,1]
        else:
            one_hot = [1,0,0]

        # Remember this move - it's a good one!
        training_data.append([prev_state, one_hot])

# Generates training data by playing games randomly, 
# but only storing the moves that improved 
# chances of winning
def generate_training_data(training_games):
    training_data = []

    for _ in range(training_games):
        prev_state = env.reset()
        
        for _ in range(goal_steps):
            # take a random action (i.e. 0=push left, 1=no push, 2=push right)
            action = env.action_space.sample()
            #env.render()

            # execute action, get feedback from env
            # will return reward=-1 for each time step
            # game is won if reach position 0.5, game is lost after 200 iterations
            # 'observation' is array of floats:
            #   position[-1.2 .. 0.6], velocity[-0.07 .. 0.07]
            new_state, reward, done, info = env.step(action) 

            # Store this action if it resulted in a beneficial move in position
            update_training_set_position(new_state, prev_state, action, training_data)
            #update_training_set_velocity(new_state, prev_state, action, training_data)

            prev_state = new_state
 
            if done:
                break

    # Save the training data in a file
    training_data_save = np.array(training_data)  
    np.save(TRAIN_DATA_FILENAME, training_data_save)

    return training_data 

def load_training_data():
    training_data = np.load(TRAIN_DATA_FILENAME)
    return training_data

# Create a NN with input layer / 1 fully-connected hidden layer / softmax output layer
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, input_size * 20, activation='tanh')
    network = fully_connected(network, (input_size * 10 + env.action_space.n * 10)/2, activation='tanh')
    network = fully_connected(network, env.action_space.n * 10, activation='tanh')

    network = fully_connected(network, env.action_space.n, activation='softmax')

    # Set optimizer, loss function, learning rate
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    # Create Deep NN model on this network
    model = tflearn.DNN(network, tensorboard_dir='log', tensorboard_verbose=3)

    return model

# Train the model (defaults to creating a new model from prev function) on the given training data
def train_model(training_data, model=False):

    # Separate the training data into inputs (X) and targets (y)
    # training_data[0]==observation, training_data[1]==one-hot action
    # So we're training the network to take the action associated with a given observation
    X = [i[0] for i in training_data]
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len (X[0]))

    # Now train the model on the inputs (X) and targets (y)
    model.fit({'input' : X}, {'targets' : y}, n_epoch=2, snapshot_step=500, show_metric=True,
        run_id='openaistuff', validation_set=0.01)

    model.save(TRAINED_MODEL_FILENAME)
    return model

def load_model(input_size):
    model = neural_network_model(input_size)
    model.load(TRAINED_MODEL_FILENAME)
    return model

# print('***Generating random training data...')
# training_data = generate_training_data(training_games)

# print('***Loading training data...')
# training_data = load_training_data()

# print('***Training model...')
# model = train_model(training_data)

print('***Loading model...')
model = load_model(input_size=2)

scores = []
choices = []
num_steps_taken = []
game_outcomes = []

print('***Running test games...')
for each_game in range(100):
    num_steps = 0
    observation = env.reset()

    for _ in range(goal_steps):
        env.render()
        action = np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])
        choices.append(action)

        observation, reward, done, info = env.step(action)
        num_steps += 1
        if done:
            break

    if( num_steps < goal_steps): 
        game_outcomes.append(1) 
    else:
        game_outcomes.append(0) 

    num_steps_taken.append(num_steps)

print(num_steps_taken)
print('Winning percentage ', sum(game_outcomes)/len(game_outcomes))
print('Average steps ', sum(num_steps_taken)/len(num_steps_taken))
print('Right: {}, No push: {}, Left: {}'.format(choices.count(2)/len(choices),choices.count(1)/len(choices),
    choices.count(0)/len(choices)))