# AI Gym CartPole-v0
# JTE 03/09/18

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
score_requirement = 50
training_games = 10000

# Here's the AI gym environment we're using
env = gym.make('CartPole-v0')

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

# Generates training data by playing games randomly, 
# but only storing the moves made in 'good' games.
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(training_games):
        score = 0
        game_memory = []
        prev_state = env.reset()
        
        for _ in range(goal_steps):
            # take a random action (i.e. apply a force of +1 or -1 to move left or right)
            action = env.action_space.sample()

            # execute action, get feedback from env
            # will return reward=1 if pole remains upright
            # if pole falls >15 degrees from vertical or moves 2.4 units from center, game ends
            # 'observation' is array of floats:
            #   cart_x_pos, cart velocity, pole angle, pole tip velocityf
            new_state, reward, done, info = env.step(action) 
            
            # Remember the prior state + the action we took in it
            game_memory.append([prev_state, action])

            prev_state = new_state
            score += reward
            if done:
                break

        # If this game gave a good score, save the score and the action/ouput tuples
        if score >= score_requirement :
            accepted_scores.append(score)

            # encode each action as 1-hot vector
            for data in game_memory: # data[0]==observation, data[1]==action
                if data[1] == 1:  
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # and record the state we were in when took this action
                training_data.append([data[0], output])

        # Regardless of whether this was a winning game, store score
        scores.append(score)

    # Save the training data in a file
    # training_data_save = np.array(training_data)  
    # np.save('saved.npy', training_data_save)

    # Print summary stats of training data
    print('Average overall score: ', mean(scores))
    print('Average accepted score: ', mean(accepted_scores))
    print(Counter(accepted_scores))

    return training_data 

# Create a NN with input layer / 1 fully-connected hidden layer / softmax output layer
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, 8, activation='relu')

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
        run_id='openaistuff', validation_set=0.05)

    return model

print('***Generating random training data...')
training_data = initial_population()

print('***Training model...')
model = train_model(training_data)

scores = []
choices = []

print('***Running test games...')
for each_game in range(20):
    score = 0
    observation = env.reset()

    for _ in range(goal_steps):
        env.render()
        action = np.argmax(model.predict(observation.reshape(-1, len(observation)))[0])
        choices.append(action)

        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break

    scores.append(score)

print(scores)
print('Average score ', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices),
    choices.count(0)/len(choices)))