#Actor Critic
import numpy as np
import tensorflow as tf
import gym

class Actor:
    def __init__(self, session, input_size, output_size, name = "main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name=  name
        
        self._build_network()
    
    def _build_network(self, h_size = 100, l_rate = 1e-2):
        with tf.variable_scope(self.net_name):
            self._keep_prob = tf.placeholder(tf.float32)
            
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1 = tf.get_variable("Actor-Weight1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            B1 = tf.get_variable("Actor-Bias1", shape=[h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            L1 = tf.nn.relu(tf.matmul(self._X, W1)+B1)
            L1 = tf.nn.dropout(L1, keep_prob=self._keep_prob)
            
            W2 = tf.get_variable("Actor-Weight2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            B2 = tf.get_variable("Actor-Bias2", shape = [h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            L2 = tf.nn.relu(tf.matmul(L1, W2)+B2)
            L2 = tf.nn.dropout(L2, keep_prob=self._keep_prob)
            
            W3 = tf.get_variable("Actor-Weight3", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            B3 = tf.get_variable("Actor-Bias3", shape=[self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            
            self._PolicyPred = tf.nn.softmax(tf.matmul(L2, W3)+B3)
        
        self._Y = tf.placeholder(shape = [None,self.output_size], dtype = tf.float32)
        
        #self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        #self._loss = tf.reduce_sum(-tf.log(self._PolicyPred)*self._Y)
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self._PolicyPred, labels = self._Y)
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._PolicyPred, feed_dict = {self._X:x, self._keep_prob:1})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack, self._keep_prob:0.6})
    
class Critic:
    def __init__(self, session, input_size, name = "main"):
        self.session = session
        self.input_size = input_size
        self.net_name=  name
        
        self._build_network()
    
    def _build_network(self, h_size = 100, l_rate = 1e-2):
        with tf.variable_scope(self.net_name):
            self._keep_prob = tf.placeholder(tf.float32)
            
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1 = tf.get_variable("Critic-Weight1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            B1 = tf.get_variable("Critic-Bias1", shape=[h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            L1 = tf.nn.relu(tf.matmul(self._X, W1)+B1)
            L1 = tf.nn.dropout(L1, keep_prob=self._keep_prob)
            
            W2 = tf.get_variable("Critic-Weight2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            B2 = tf.get_variable("Critic-Bias2", shape = [h_size], initializer=tf.contrib.layers.xavier_initializer())
            
            L2 = tf.nn.relu(tf.matmul(L1, W2)+B2)
            L2 = tf.nn.dropout(L2, keep_prob=self._keep_prob)
            
            W3 = tf.get_variable("Critic-Weight3", shape=[h_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            B3 = tf.get_variable("Critic-Bias3", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
            
            self._Vpred = tf.matmul(L2, W3)+B3
        
        self._Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)
        
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Vpred))
        self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Vpred, feed_dict = {self._X:x, self._keep_prob:1})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack, self._keep_prob:0.6})
    
env = gym.make("SpaceInvaders-ram-v0")

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9

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
        actor = Actor(sess, input_size, output_size)
        critic = Critic(sess, input_size)
        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            step_count = 0
            tot_score = 0
            done = False
            state = env.reset().reshape(1,input_size)
            actor_input = np.zeros((1,output_size))
            
            while not done:
                action = np.argmax(actor.predict(state))
                
                next_state, reward, done, _ = env.step(action)
                next_state = next_state
                NextVal = critic.predict(next_state)
                
                step_count += 1
                tot_score += reward

                if done:
                    reward = -1000
                    
                UpdateVal = reward + NextVal * dis
                actor_input[0,action] = UpdateVal - critic.predict(state)
                
                actor.update(state, actor_input)
                critic.update(state, UpdateVal)
                
                state = next_state.reshape(1,input_size)
                env.render()
            
            print("Episode {}: Score - {}".format(episode, int(tot_score)))
            episode += 1
                
        bot_play(actor)

main()