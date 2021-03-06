"""
Author: Muhammad Umar Amanat
Viterbi implementation is from "Python machine learning by Nishant Shukla with kenneth Fricklas books"
"""
import numpy as np
import tensorflow as tf

class HMM(object):
  def __init__(self, initial_prob, trans_prob, obs_prob):
    """ 
    intial_prob: initial proabilities of states, vector
    trans_prob: transition probabilities, matrix
    obj_prob: observed probabilities, matrix
    """
    self.N = np.size(initial_prob)
    self.initial_prob = initial_prob
    self.trans_prob = trans_prob
    self.emission = tf.constant(obs_prob) # emission probabilities matrix
    
    # check dimension in order to flee from tensorflow error
    assert self.initial_prob.shape == (self.N, 1)
    assert self.trans_prob.shape == (self.N, self.N)
    assert obs_prob.shape[0] == self.N 
    
    self.obs_idx = tf.placeholder(tf.int32, name='observation_id')
    self.fwd = tf.placeholder(tf.float64, name='forward')
    self.viterbi = tf.placeholder(tf.float64, name='viterbi')
    
  def get_emission(self, obs_id):
    slice_loc = [0, obs_id]
    num_rows = self.emission.shape[0]
    return tf.slice(self.emission, slice_loc, [num_rows, 1])
    
  # just for initialization of fwd 
  def forward_init_op(self):
    obs_prob = self.get_emission(self.obs_idx)
    fwd = tf.multiply(self.initial_prob, obs_prob)
    return fwd
   
  # def forward_op(self):
  #   transition = tf.matmul(self.fwd, 
  #                          tf.transpose(self.get_emission(self.obs_idx)))
  #   weighted_transitions = transition * self.trans_prob
  #   fwd = tf.reduce_sum(weighted_transitions, 0)
  #   return tf.reshape(fwd, tf.shape(self.fwd))

  def backpt_op(self):
    back_transitions = tf.matmul(self.viterbi, np.ones((1, self.N)))
    weighted_back_transitions = back_transitions * self.trans_prob
    return tf.argmax(weighted_back_transitions, 0)

  def decode_op(self):
    transitions = tf.matmul(self.viterbi, 
                            tf.transpose(self.get_emission(self.obs_idx)))
    weighted_transitions = transitions * self.trans_prob
    viterbi = tf.reduce_max(weighted_transitions, 0)
    return tf.reshape(viterbi, tf.shape(self.viterbi))
    
# driver function for Viterbi algorithm
def viterbi_decode(sess, hmm, observations):
  viterbi = sess.run(hmm.forward_init_op(), feed_dict={hmm.obs_idx: 
                                                       observations[0]})
  backpts = np.ones((hmm.N, len(observations)), 'int32') * -1
  for t in range(1, len(observations)):
    viterbi, backpt = sess.run([hmm.decode_op(), hmm.backpt_op()], 
                               feed_dict={hmm.obs_idx: observations[t], 
                                          hmm.viterbi: viterbi})
    backpts[:, t] = backpt
  tokens = [viterbi[:, -1].argmax()]
  for i in range(len(observations) - 1, 0, -1):
    tokens.append(backpts[tokens[-1], i])
  return tokens[::-1]
    
if __name__ == '__main__':
  initial_prob = np.array([[0.6], 
                           [0.4]])
  trans_prob = np.array([[0.7, 0.3],
                         [0.4, 0.6]])
  obs_prob = np.array([[0.1, 0.4, 0.5],
                       [0.6, 0.3, 0.1]])
  hmm = HMM(initial_prob=initial_prob, trans_prob=trans_prob, 
            obs_prob=obs_prob)
  observations = [0, 2, 0, 2, 1]
  with tf.Session() as sess:
    # prob = forward_algorithm(sess, hmm, observations)
    # print('Probability of observing {} is {}'.format(observations, prob))
    seq = viterbi_decode(sess, hmm, observations)
    print('Most likely hidden states are {}'.format(seq))