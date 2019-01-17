#observable states
obs = ('normal', 'cold', 'dizzy')
#hidden states: for which we need to predict
states = ('Healthy', 'Fever')

#start prob of hidden states(dataset: target distibution)
start_p = {'Healthy': 0.6, 'Fever': 0.4}
#Internal transition probs of target states
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
#Emition probs of observations for target states
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }

#It takes input as: 
# sequesnce of observations for which we need to predict,
# target states,
# start_probs of target states,
# trans_probs of target states internally
# emition_probs of observations for each target state
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    
    print "\nTime Step t = 0 curr_obs:" + obs[0] + "\n\n"
    # for time step t = 0
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        print "For State:"+ st + "->[start_p["+st+"] * emit_p["+st+"]["+obs[0]+"]]"
        print V[0][st]
        
    for line in dptable(V):
        print line
        
        
    print "\n\nTime Step t>0\n"
    # Run Viterbi when time step t > 0
    for t in range(1, len(obs)):
        V.append({})
        print "\nTime Step t = " + (str)(t) + "  curr_obs:" + obs[t]
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            print "\nTransProb = prev_state(obs)_prob * trans_p(prev_state(any)->cur_state)" 
            print "CurState:" + st + " from prev state[H/F]  maxTransProb: " + (str)(max_tr_prob)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    print "For State:"+ st + "->[max_tr_prob:"+prev_st+"->"+st+" * emit_p["+st+"]["+obs[t]+"]]"
                    print V[t][st]
                    break
    
    print "\n\nEach time step[obs seq] prob of patient's state"
    for line in dptable(V):
        print line
    
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

        
viterbi(obs, states, start_p, trans_p, emit_p)

'''
Time Step t = 1  curr_obs:cold

TransProb = prev_state(obs)_prob * trans_p(prev_state(any)->cur_state)
CurState:Healthy from prev state[H/F]  maxTransProb: 0.21
For State:Healthy->[max_tr_prob:Healthy->Healthy * emit_p[Healthy][cold]]
{'prob': 0.084, 'prev': 'Healthy'}

TransProb = prev_state(obs)_prob * trans_p(prev_state(any)->cur_state)
CurState:Fever from prev state[H/F]  maxTransProb: 0.09
For State:Fever->[max_tr_prob:Healthy->Fever * emit_p[Fever][cold]]
{'prob': 0.027, 'prev': 'Healthy'}

Time Step t = 2  curr_obs:dizzy

TransProb = prev_state(obs)_prob * trans_p(prev_state(any)->cur_state)
CurState:Healthy from prev state[H/F]  maxTransProb: 0.0588
For State:Healthy->[max_tr_prob:Healthy->Healthy * emit_p[Healthy][dizzy]]
{'prob': 0.00588, 'prev': 'Healthy'}

TransProb = prev_state(obs)_prob * trans_p(prev_state(any)->cur_state)
CurState:Fever from prev state[H/F]  maxTransProb: 0.0252
For State:Fever->[max_tr_prob:Healthy->Fever * emit_p[Fever][dizzy]]
{'prob': 0.01512, 'prev': 'Healthy'}


Each time step[obs seq] prob of patient's state
           0            1            2
Healthy: 0.30000 0.08400 0.00588
Fever: 0.04000 0.02700 0.01512
The steps of states are Healthy Healthy Fever with highest probability of 0.01512
'''
