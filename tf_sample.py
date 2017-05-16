from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import os
from six.moves import cPickle
from tf_model import Model
import random
from six import text_type
from animation import animate_action
import check_mat


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--action_type', type=str, default='shoot_bow',
                        help='action type')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory containing data')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    full_action_dict, actions_pruned = check_mat.save_actions(args.data_dir + '/joint_positions')
    data = check_mat.fetch_action_data(args.action_type, full_action_dict, actions_pruned)
    sample(args, data)


def sample(args, data):
    length = len(data[0]['scale'][0])
    res = np.zeros((2, 15, length)) #TEMP
    for i in range(30): #This corresponds to each x, y of each joint 
        with open(os.path.join(args.save_dir, 'config' + str(i) + '.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(args.save_dir, 'pos_dict' + str(i) + '.pkl'), 'rb') as f:
            chars, vocab = cPickle.load(f)
        model = Model(saved_args, training=False)
        truth = random.choice(data)['pos_world']
        primes = truth[:,:,0]
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                prime = primes[i%2, i//2]
                res[i%2, i//2, :] = model.sample(sess, chars, vocab, length, prime,
                                   args.sample)

    #Animate the final result 
    animate_action(res, "prediction")
    animate_action(truth, "truth")

if __name__ == '__main__':
    main()
