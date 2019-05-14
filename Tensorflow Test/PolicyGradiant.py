#reinforce

import numpy as np
import tensorflow as tf
import gym

class PolicyNet:
    def __init__(self, session, input_size, output_size, name = "main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name=  name
        
        self._build_network()
    
    def _build_network(self, h_size = 100, l_rate = 1e-1):
        with tf.variable_scope(self.net_name):
            self._keep_prob = tf.placeholder(tf.float32)
            
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1 = tf.get_variable("Weight1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            B1 = tf.get_variable("Bias1", shape=[h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            L1 = tf.nn.relu(tf.matmul(self._X, W1)+B1)
            L1 = tf.nn.dropout(L1, keep_prob=self._keep_prob)
            
            W2 = tf.get_variable("Weight2", shape=[h_size, 10], initializer=tf.contrib.layers.xavier_initializer())
            B2 = tf.get_variable("Bias2", shape = [10], initializer=tf.contrib.layers.xavier_initializer())
            
            L2 = tf.nn.relu(tf.matmul(L1, W2)+B2)
            L2 = tf.nn.dropout(L2, keep_prob=self._keep_prob)
            
            W3 = tf.get_variable("Weight3", shape=[10, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            B3 = tf.get_variable("Bias3", shape = [self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            
            self._PolicyPred = tf.nn.softmax(tf.matmul(L2, W3)+B3)
        
        self._Y = tf.placeholder(shape = [None,self.output_size], dtype = tf.float32)
        
        #self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._loss = tf.reduce_sum(-tf.log(self._PolicyPred)*self._Y)
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._PolicyPred, feed_dict = {self._X:x, self._keep_prob:1})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack, self._keep_prob:0.6})
    
env = gym.make("Breakout-ram-v0")

learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.95

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
    max_episodes = 1000
    
    with tf.Session() as sess:
        pnet = PolicyNet(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            e = 1/((episode//10)+1)
            step_count = 0
            tot_score = 0
            rewardstep = 0
            R_traj = np.empty(0).reshape(0,1)
            a_traj = np.empty(0).reshape(0,output_size)
            x_stack = np.empty(0).reshape(0, input_size)
            dis_factor = np.empty(0).reshape(0,1)
            done = False
            state = env.reset()
            
            while not done:
                dis_factor = np.vstack([dis**step_count, dis_factor])
                
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(pnet.predict(state))
                next_state, reward, done, _ = env.step(action)
                step_count += 1
                tot_score += reward
                
                if done:
                    reward = -10
                    
                if (reward == 1) and (step_count-rewardstep <= 100):
                    rewardstep = step_count
                    reward *= 2
                
                x_stack = np.vstack([x_stack, state])
                
                R_traj = np.vstack([R_traj, 0])
                R_traj += dis_factor*reward
                a_traj = np.vstack([a_traj, np.eye(output_size)[action]])
                
                state = next_state
                step_count += 1
                
            R_stack = R_traj * a_traj
            
            print("Episode {}: Score - {}".format(episode, tot_score))
            pnet.update(x_stack, R_stack)
            episode += 1
            
            '''
            if episode % 30 == 29:
                for _ in range(10):
                    minibatch = random.sample(replay_buffer, 100)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
                print("loss: ", loss)
            '''
                
        bot_play(pnet)

main()