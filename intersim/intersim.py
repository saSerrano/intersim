import os
import numpy as np
from copy import copy,deepcopy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils

from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.core import Core
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter, LinearParameter
from mushroom_rl.utils.callbacks import CollectDataset, CollectParameters

class DecayParameter(Parameter):
    r"""
    This class implements a decaying parameter. The decay follows the formula:

    .. math::
        v_n = \dfrac{v_0}{n^p}

    where :math:`v_0` is the initial value of the parameter,  :math:`n` is the number of steps and  :math:`p` is an
    arbitrary exponent.

    """
    def __init__(self, value, exp=1., min_value=None, max_value=None, size=(1,)):
        """
        Constructor.

        Args:
            value (float): initial value of the parameter;
            exp (float, 1.): exponent for the step decay;
            min_value (float, None): minimum value that the parameter can reach when decreasing;
            max_value (float, None): maximum value that the parameter can reach when increasing;
            size (tuple, (1,)): shape of the matrix of parameters; this shape can be used to have a single parameter for
                each state or state-action tuple.

        """
        self._exp = exp

        super().__init__(value, min_value, max_value, size)

        self._add_save_attr(_exp='primitive')

    def _compute(self, *idx, **kwargs):
        n = np.maximum(self._n_updates[idx], 1)

        return self._initial_value / n ** self._exp

# Print Q-table
# shape = agent.Q.shape
# q = np.zeros(shape)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         state = np.array([i])
#         action = np.array([j])
#         q[i, j] = agent.Q.predict(state, action)
# print(q)

def meanEpisodicReward(data):
    '''
    DESCRIPTION

    Computes the mean episodic total reward over a data
    set of samples.

    INPUT

    data (list-tuple): list of tuples, each representing
    an RL sample in the mushroomRL style: state, action,
    reward, next_state, absorbing, last.

    OUTPUT

    mean_r (float): mean episodic reward.
    stdv_r  (float): standard deviation of the mean epi-
    sodic reward.
    '''
    # Get the total reward of each episode
    episodic_r = []
    current_r = []
    for i in range(len(data)):
        sample = data[i]
        current_r.append(sample[2])
        # Last sample from the current episode
        if sample[-1] or i == (len(data)-1):
            episodic_r.append(np.sum(current_r))
            current_r = []
    
    # Compute mean and st-dev of total reward over the data set
    tmp = np.array(episodic_r)
    mean_r = float(np.mean(tmp))
    stdv_r  = float(np.std(tmp))

    return mean_r, stdv_r

def plotData(mean_data,stdv_data,x_data,x_axis_name,y_axis_name,fname,smooth_window=0,show=False):
    '''
    DESCRIPTION

    Renders the mean episodic reward of an RL agent.

    INPUT

    mean_r (list-float): mean episodic reward.
    stdv_r (list-float): episodic reward standard deviation.
    steps    (list-int): environmenment steps.
    output_dir    (str): directory where the plot will be saved.
    smooth_window (int): length of the smoothing window.
    show         (bool): whether the plot is shown after render.
    '''
    assert len(mean_data) == len(stdv_data) and len(mean_data) == len(x_data)
    
    # Preprocess data
    x, y = np.array(x_data), np.array(mean_data)
    upper, lower = y + np.array(stdv_data), y - np.array(stdv_data)
    if smooth_window > 0:
        y = utils.simpleAvgSmooth(y,smooth_window)
        upper = utils.simpleAvgSmooth(upper,smooth_window)
        lower = utils.simpleAvgSmooth(lower,smooth_window)
    
    # Plot data
    fig, ax = plt.subplots()
    ax.plot(x, y,color='g')
    ax.fill_between(x,upper,lower,color='g',alpha=0.3)
    ax.set(xlabel=x_axis_name, ylabel=y_axis_name)
    ax.grid()
    plt.tight_layout()

    # Save graph
    fig.savefig(fname)

    # Show graph
    if show:
        plt.show()

def rl(parameters):
    '''
    Perform Q-learning

    Check the following link for an example:
    https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/grid_world_td.py
    
    '''
    # Get training parameters
    params = None
    assert isinstance(parameters,dict) or isinstance(parameters,str)
    if isinstance(parameters,dict):
        params = deepcopy(parameters)
    else:
        params = utils.loadJson(parameters)
    task_name = params['task_name']
    horizon = params['horizon']
    gamma = params['gamma']
    training_episodes = params['training_episodes']
    n_training_episodes_per_eval = params['n_training_episodes_per_eval']
    training_steps = params['training_steps']
    n_training_steps_per_eval = params['n_training_steps_per_eval']
    epsilon_init = params['epsilon_init']
    epsilon_final = params['epsilon_final']
    learning_rate_init = params['learning_rate_init']
    learning_rate_final = params['learning_rate_final']
    final_value_time = params['final_value_time']
    save_data = params['save_data']
    output_dir = params['output_dir']

    # Determine how training duration is specified
    assert (isinstance(training_episodes,int) and training_steps == None) or (training_episodes == None and isinstance(training_steps,int))
    training_duration_type, duration, period = '', 0, 0
    if isinstance(training_episodes,int):
        training_duration_type = 'episodes'
        duration = training_episodes
        period = n_training_episodes_per_eval
    else:
        training_duration_type = 'steps'
        duration = training_steps
        period = n_training_steps_per_eval

    # Start random seed gen.
    np.random.seed()

    # Create output directory
    utils.makeDir(output_dir)
    assert os.path.isdir(output_dir)

    # RL environment
    mdp = Gym(task_name, horizon, gamma)

    # Specify the changing period for linear parameters
    change_period = int(duration*horizon*final_value_time) if training_duration_type == 'episodes' else int(duration*final_value_time)

    # Policy
    epsilon = None
    if isinstance(epsilon_init,float) and isinstance(epsilon_final,float):
        epsilon = LinearParameter(value=epsilon_init, threshold_value=epsilon_final, n=change_period)
    else:
        epsilon = Parameter(value=epsilon_init)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    lr = None
    if isinstance(learning_rate_init,float) and isinstance(learning_rate_final,float):
        lr = LinearParameter(value=learning_rate_init, threshold_value=learning_rate_final, n=change_period)
    else:
        lr = Parameter(learning_rate_init)
    algorithm_params = dict(learning_rate=lr)
    agent = QLearning(mdp.info, pi, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    collect_epsilon = CollectParameters(epsilon)
    collect_lr      = CollectParameters(lr)
    callbacks = [collect_dataset,collect_epsilon,collect_lr]
    core = Core(agent, mdp, callbacks)

    # Training loop
    eval_mean_reward = []
    eval_stdv_reward = []
    eval_steps  = []
    n_iter, n_res = duration // period, duration % period
    for i in range(n_iter):        
        # Train
        training_period = period + n_res if i == n_iter-1 else period
        if training_duration_type == 'episodes':
            core.learn(n_episodes=training_period, n_steps_per_fit=1, render=False, quiet=True, record=False)
        else:
            core.learn(n_steps=training_period, n_steps_per_fit=1, render=False, quiet=True, record=False)
        
        # Evaluate
        eval_data = core.evaluate(n_episodes=10, render=False, quiet=True, record=False)
        mean_r, stdv_r = meanEpisodicReward(eval_data)
        eval_mean_reward.append(mean_r)
        eval_stdv_reward.append(stdv_r)
        eval_steps.append(len(collect_dataset.get()))

    # Get recollected data
    data = collect_dataset.get()
    eps_history = collect_epsilon.get()
    lr_history = collect_lr.get()

    # Save training data
    if save_data:
        utils.saveDataset(data,os.path.join(output_dir,'dataset.csv'),True,True)

    # Save the trained agent and parameters file
    core.agent.save(os.path.join(output_dir,'agent.msh'),True)
    utils.saveJson(params,os.path.join(output_dir,'parameters.json'))

    # Plot the agent's performance and parameters
    smooth_window = 9
    eval_fname = os.path.join(output_dir,'evaluation.png')
    eps_fname  = os.path.join(output_dir,'exploration_rate.png')
    lr_fname   = os.path.join(output_dir,'learning_rate.png')
    plotData(eval_mean_reward,eval_stdv_reward,eval_steps,'Env. Steps','Episodic Total Reward',eval_fname,smooth_window,False)
    plotData(eps_history,[0.0 for _ in eps_history],[i+1 for i in range(len(eps_history))],'Env. Steps','Exploration Rate',eps_fname,0,False)
    plotData(lr_history ,[0.0 for _ in lr_history],[i+1 for i in range(len(lr_history))],'Env. Steps','Learning Rate',lr_fname,0,False)

def cluster(m,n_clusters,mode):
    '''
    Cluster the elements in each row (or column) and return the assigned labels.
    '''
    assert len(m.shape) == 2 and n_clusters >= 2 and mode in ['rows','columns']
    
    a = None
    if mode == 'rows':
        a = np.copy(m)
    else:
        a = np.transpose(np.copy(m))

    labels = np.zeros(a.shape,dtype=np.int64)
    for i in range(a.shape[0]):
        # Cluster row elements
        x = np.copy(a[i,:]).reshape((a.shape[1],1))
        km = KMeans(n_clusters=n_clusters, random_state=0).fit(x)

        # Sort clusters by their centroid
        l, c = km.labels_, km.cluster_centers_.flatten()
        cluster_ids = [(j,c[j]) for j in range(c.shape[0])].sort(key=lambda y : y[1])
        id_map = {}
        for j in range(len(cluster_ids)):
            id_map[cluster_ids[0]] = j
        for j in range(a.shape[1]):
            labels[i,j] = id_map[l[j]]
    
    if mode == 'rows':
        return labels
    else:
        return np.transpose(labels)

def label_histograms(m,n_labels,mode):
    '''
    Builds a matrix in which each row represents the histogram
    of labels assigned in the rows or columns of input matrix m.

    INPUT

    m (np-array): input matrix.
    n_labels (int): number of different labels present in 'm'.
    mode (str): string indicating wheather labels should counted
    over rows or columns.

    OUTPUT

    f (np-array): matrix containing the label histograms (one in
    each row).
    '''
    assert len(m.shape) == 2 and n_labels >= 2 and mode in ['rows','columns']
    a = None
    if mode == 'rows':
        a = np.copy(m)
    else:
        a = np.transpose(np.copy(m))

    f = np.zeros((a.shape[0],n_labels),dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(n_labels):
            f[i,j] = float(a[i,:].tolist().count(j)) / float(a.shape[1])
    
    return f

def intersection_mats(q1,q2,n_clusters=2):
    '''
    DESCRIPTION

    This function computes the action and state intersection
    matrices for a pair of Q-tables.

    INPUT

    q1 (np-array): q-table 1.
    q2 (np-array): q-table 2.
    n_clusters (int): number of clusters.

    OUTPUT

    inter_a (np-array): actions intersection matrix.
    inter_s (np-array): states intersection matrix.
    '''
    # Compute the frequency matrices 
    fa1 = label_histograms(cluster(q1,n_clusters,'rows'),n_clusters,'columns')
    fs1 = label_histograms(cluster(q1,n_clusters,'columns'),n_clusters,'rows')
    fa2 = label_histograms(cluster(q2,n_clusters,'rows'),n_clusters,'columns')
    fs2 = label_histograms(cluster(q2,n_clusters,'columns'),n_clusters,'rows')
    
    # Compute intersection matrices
    inter_a, inter_s = np.zeros((fa1.shape[0],fa2.shape[0]),dtype=np.float64), np.zeros((fs1.shape[0],fs2.shape[0]),dtype=np.float64)
    for i in range(inter_a.shape[0]):
        for j in range(inter_a.shape[1]):
            # Intersection of histograms
            tmp = np.concatenate((np.copy(fa1[i,:]).reshape(1,n_clusters),np.copy(fa2[i,:]).reshape(1,n_clusters)),axis=0)
            inter_a[i,j] = np.sum(np.min(tmp,axis=0))
    for i in range(inter_s.shape[0]):
        for j in range(inter_s.shape[1]):
            # Intersection of histograms
            tmp = np.concatenate((np.copy(fs1[i,:]).reshape(1,n_clusters),np.copy(fs2[i,:]).reshape(1,n_clusters)),axis=0)
            inter_s[i,j] = np.sum(np.min(tmp,axis=0))

    return inter_a, inter_s

def intertask_similarity(inter_a,inter_s):
    '''
    DESCRIPTION

    Computes the mean/frobenius-based similarity scores
    of two Q-tables, given their action and state inter-
    section matrices.

    INPUT

    inter_a (np-array): action intersection matrix.
    inter_s (np-array): state intersection matrix.

    OUTPUT

    mean_a (np-array): mean-based action similarity score.
    mean_s (np-array): mean-based state similarity score.
    frob_a (np-array): frobenius-based action similarity score.
    frob_s (np-array): frobenius-based state similarity score.
    '''
    mean_a = float(np.mean(inter_a))
    mean_s = float(np.mean(inter_s))
    frob_a = np.linalg.norm(inter_a,'fro')
    frob_s = np.linalg.norm(inter_s,'fro')
    return mean_a, mean_s, frob_a, frob_s

def transfer(src_q,inter_a,inter_s):
    '''
    DESCRIPTION

    Computes the Q-table for the target task, trans-
    ferred from the source task Q-table.

    INPUT

    src_q   (np-array): source task Q-table.
    inter_a (np-array): action intersection matrix.
    inter_s (np-array): state intersection matrix.

    OUTPUT

    tgt_q   (np-array): transferred target task Q-table.
    '''
    # Select the most similar source state-action pair to each
    # target state-action pair
    tgt_q = np.zeros((inter_s.shape[1],inter_a.shape[1]),dtype=np.float64)
    for i in range(tgt_q.shape[0]):
        ms_state = np.argmax(inter_s[:,i])
        for j in range(tgt_q.shape[1]):
            ms_action = np.argmax(inter_a[:,j])
            tgt_q[i,j] = copy(src_q[ms_state,ms_action])
    
    return tgt_q

def main():
    print(0)

    # # Learn Q tables
    # q1 = rl('task 1')
    # q2 = rl('task 2')

    # # Compute instersection matrices
    # n_clusters = 2
    # inter_a, inter_s = intersection_mats(q1,q2,n_clusters)

    # # Measure similarity scores
    # ma,ms,fa,fs = intertask_similarity(inter_a,inter_s)

    # # Transfer Q-values
    # tl_q2 = transfer(q1,inter_a,inter_s)

if __name__ == '__main__':
    # main()
    params = {}
    params['task_name'] = 'FrozenLake-v1'
    params['horizon'] = 100
    params['gamma'] = 0.95
    params['training_episodes'] = None
    params['n_training_episodes_per_eval'] = 100
    params['training_steps'] = 10000
    params['n_training_steps_per_eval'] = 1000
    params['epsilon_init'] = 1.0
    params['epsilon_final'] = 0.1
    params['learning_rate_init'] = 0.10
    params['learning_rate_final'] = 0.01
    params['final_value_time'] = 0.5
    params['save_data'] = False
    params['output_dir'] = '/home/sergio/code/intersim/tmp'
    rl(params)