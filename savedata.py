# encoding: utf-8
'''
@project = _project_name_
@file=
@author = wangchen
@create_time = $
'''


import numpy as np
import matplotlib.pyplot as plt


def save_result(f, episode, loss, score):
    f.write('Episode: ' + str(episode) +
            ' Loss: ' + str(loss) +
            ' score: ' + str(score) + '\n')


def get_data(data_path):
    f = open(data_path, 'r')
    episode = []
    loss = []
    score = []
    for line in f:
        line = line.strip('\n')
        newline = line.split()
        episode.append(int(newline[1]))
        loss.append(float(newline[3]))
        score.append(float(newline[5]))
    return episode, loss, score


def analyse_data(ep, loss, score):
    ep = np.resize(np.array(ep), [len(ep)//100, 100])
    loss_Mat = np.resize(np.array(loss), [len(loss)//100, 100])
    score_Mat = np.resize(np.array(score), [len(score)//100, 100])
    loss_mean = np.average(loss_Mat, 1)
    score_mean = np.average(score_Mat, 1)
    ep = ep[:, 0]
    return ep, loss_mean, score_mean


if __name__ == '__main__':
    dqns_ep, dqns_loss, dqns_score = get_data("./dqns_results.txt")
    dqn_ep, dqn_loss, dqn_score = get_data("./dqn_results.txt")
    dqns_ep, dqns_loss, dqns_score = analyse_data(dqns_ep, dqns_loss, dqns_score)
    dqn_ep, dqn_loss, dqn_score = analyse_data(dqn_ep, dqn_loss, dqn_score)
    plt.figure(num=1, figsize=(8, 16))
    plt.subplot(211)
    plt.plot(dqns_ep, dqns_loss, label='DQNs', color='red', linewidth=1, linestyle='-')
    plt.plot(dqn_ep, dqn_loss, label='DQN', color='blue', linewidth=1, linestyle='-')
    plt.subplot(212)
    plt.plot(dqns_ep, dqns_score, label='DQNs', color='red', linewidth=1, linestyle='-')
    plt.plot(dqn_ep, dqn_score, label='DQN', color='blue', linewidth=1, linestyle='-')
    plt.show()
