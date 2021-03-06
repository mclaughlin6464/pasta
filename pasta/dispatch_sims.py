'''
Send a collection of models to the cluster.
'''
import os
from subprocess import call
import numpy as np

outputdir = '/u/ki/swmclau2/des/statmech/full2/'
max_time = 6

def make_kils_command(jobname,N,B_goal, xb, xp, n_steps, max_time = max_time, outputdir = outputdir, queue='kipac-ibq'):#'bulletmpi'):
    '''
    Return a list of strings that comprise a bash command to call trainingHelper.py on the cluster.
    Designed to work on ki-ls's batch system
    :param jobname:
        Name of the job. Will also be used to make the parameter file and log file.
    :params N, B_goal, xb, xp, n_steps:
        Input params for charged_ising.py
    :param max_time:
        Time for the job to run, in hours.
    :param outputdir:
        Directory to store output and param files.
    :param queue:
        Optional. Which queue to submit the job to.
    :return:
        Command, a list of strings that can be ' '.join'd to form a bash command.
    '''
    log_file = jobname + '.out'
    command = ['bsub',
               '-q', queue,
               '-n', str(1),
               '-J', jobname,
               '-oo', os.path.join(outputdir, log_file),
               '-W', '%d:00' % max_time,
               'python', os.path.join('/u/ki/swmclau2/Git/pasta/pasta/', 'charged_ising.py'),
                str(N), str(B_goal), str(xb), str(xp), str(n_steps), outputdir]

    return command

xbs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0]
xps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

N = 30
#B_goal = 100 
n_steps = 100000
'''
for xb in xbs:
    for xp in xps:
        jobname = 'xb_%0.2f_xp_%0.2f'%(xb,xp)
        command = make_kils_command(jobname, N,B_goal, xb, xp, n_steps)

        call(command)

'''
B_goals = np.logspace(-1, 1, 10)
for xb in xbs:
    #if xb != 0.1:
    #    continue
    for xp in xps:
        #if xp != 0.2:
        #    continue
        for B_goal in B_goals:
            #if np.abs(B_goal - 0.464)> 1e-3:
            #    continue
            jobname = 'xb_%0.2f_xp_%0.2f_B_g%.2e'%(xb,xp,B_goal)
            command = make_kils_command(jobname, N,B_goal, xb, xp, n_steps,max_time, os.path.join(outputdir, jobname))

            call(command)
