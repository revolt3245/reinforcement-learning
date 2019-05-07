import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque

class DQN:
    def __init__(self, session, input_size, output_size, name = "main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name=  name
        
        self._build_network()
    
    def _build_network(self, h_size = 10, l_rate = 1e-1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1 = tf.get_variable("Weight1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            
            W2 = tf.get_variable("Weight2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            
            self._Qpred = tf.matmul(layer1, W2)
        
        self._Y = tf.placeholder(shape = [None,self.output_size], dtype = tf.float32)
        
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X:x})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001

learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)
    
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        
        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + dis * np.max(DQN.predict(next_state))
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
    return DQN.update(x_stack, y_stack)

def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done,_ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
        
def main():
    max_episodes = 2000
    replay_buffer = deque()
    
    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            e = 1/((episode//10)+1)
            done = False
            step_count = 0
            
            state = env.reset()
            
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                    
                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100
                
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                    
                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
            print("Episode: {}   step : {}".format(episode, step_count))
            if step_count > 10000:
                pass
            
            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)
                print("Loss: ", loss)
        bot_play(mainDQN)
        
main()

'''
X = tf.placeholder(tf.float32, [None, input_size], name = "input_x")
W1 = tf.get_variable("W1", shape = [input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W1)

Y = tf.placeholder(shape = [None, output_size], dtype = tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

num_episodes = 2000
dis = 0.9
rList = []
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1./((i/10) + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False
    
    while not done:
        env.render()
        step_count += 1
        x = np.reshape(s, [1, input_size])
        Qs = sess.run(Qpred, feed_dict = {X: x})
        
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
        
        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0,a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = sess.run(Qpred, feed_dict = {X:x1})
            Qs[0,a] = reward + dis*np.max(Qs1)
        
        sess.run(train, feed_dict = {X:x, Y:Qs})
        s = s1
    rList.append(step_count)
    if (len(rList) > 10) and (np.mean(rList[-10:]) > 500):
        break
    
observation = env.reset()
reward_sum = 0
while True:
    env.render()
    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict = {X:x})
    a = np.argmax(Qs)
    
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
    
env.close()

'''