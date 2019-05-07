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
        self.net_name=name
        
        self._build_network()
        
    def _build_network(self, imagesize = [210,160,3], l_rate = 1e-1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size])
            X_imag = tf.reshape(self._X, [-1]+imagesize)
            
            W1 = tf.Variable(tf.random_normal([11,11,3,9]))
            L1 = tf.nn.conv2d(X_imag, W1, strides = [1,1,1,1], padding = 'SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize = [1,5,5,1], strides = [1,5,5,1], padding = 'SAME')
            
            W2 = tf.Variable(tf.random_normal([11,11,9,6]))
            L2 = tf.nn.conv2d(L1, W2, strides = [1,1,1,1], padding = 'SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
            
            L2 = tf.reshape(L2, [-1, 21*16*6])
            W3 = tf.get_variable(shape = [21*16*6, 100], initializer = tf.contrib.layers.xavier_initializer(), name = "W3")
            L3 = tf.matmul(L2, W3)
            L3 = tf.nn.tanh(L3)
            
            W4 = tf.get_variable(shape = [100, self.output_size], initializer = tf.contrib.layers.xavier_initializer(), name = "W4")
            L4 = tf.matmul(L3, W4)
            self._Qpred = tf.nn.tanh(L4)
            
        self._Y = tf.placeholder(shape = [None, self.output_size], dtype = tf.float32)
        
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X:x})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
    
env = gym.make("Breakout-v0")

learning_rate = 1e-1
i_size = env.observation_space.shape

input_size = 1
for i in i_size:
    input_size *= i
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 5000

def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
        
    return op_holder
    
def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis*np.max(targetDQN.predict(next_state))
            
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state.reshape(210*160*3)])
        
    return mainDQN.update(x_stack, y_stack)

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
    max_episodes = 5000
    replay_buffer = deque()
    
    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name = "main")
        targetDQN = DQN(sess, input_size, output_size, name = "target")
        tf.global_variables_initializer().run()
        
        copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
        
        sess.run(copy_ops)
        
        for episode in range(max_episodes):
            e = 1/((episode//10)+1)
            done = False
            state = env.reset()
            totreward = 0
            
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                    
                next_state, reward, done, _ = env.step(action)
                
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                    
                state = next_state
                totreward += reward
            print("Episode {}: Score - {}".format(episode, totreward))
            
            if episode % 10 == 1:
                for _ in range(2):
                    minibatch = random.sample(replay_buffer, 30)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
                print("loss: ", loss)
                sess.run(copy_ops)
                
        bot_play(mainDQN)
        
main()