#!/bin/env python
'''
   SimpleBool is a python package for dynamic simulations of boolean network 
    Copyright (C) 2013  Junyan Lu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division
from collections import defaultdict
import os
import array
import cPickle
import networkx as nx
import numpy as np
from random import randint,shuffle
import sys

# global variables. RegNodes record regulators for each node. TruthTab record the truth table for each node
global RegNodes, TruthTab

def ParaParser(ParaFile):  
    '''#parser parameters for simulation and transition matrix building'''
    INPUT = {'rules'    :    'rules.txt',
             'ini_on'   :    '',
             'ini_off'  :    '',
             'turn_on'  :    '',
             'turn_off' :    '',
             'rounds'   :    1,
             'steps'    :    1,
             'mode'     :    'Sync',
             'plot_nodes' :  '',
             'trans_mode' :  0,
             'initial_limit' : 0
            }  # define parameters

    for each_line in open(ParaFile).readlines():
        para_name = each_line.split('=')[0].strip()
        para_value = each_line.split('=')[1].strip()
        if para_name in INPUT.keys():
            INPUT[para_name] = para_value
        else:
            print "Error: Unknown Parameters: %s" % para_name
             
    # formalize parameters

    try:
        INPUT['rules'] = str(INPUT['rules'])
        INPUT['ini_on'] = [node.strip() for node in INPUT['ini_on'].split(',')]
        INPUT['ini_off'] = [node.strip() for node in INPUT['ini_off'].split(',')]
        INPUT['turn_on'] = [node.strip() for node in INPUT['turn_on'].split(',')]
        INPUT['turn_off'] = [node.strip() for node in INPUT['turn_off'].split(',')]
        INPUT['plot_nodes'] = [node.strip() for node in INPUT['plot_nodes'].split(',')]
        INPUT['rounds'] = int(INPUT['rounds'])
        INPUT['steps'] = int(INPUT['steps'])
        INPUT['mode'] = str(INPUT['mode'])
        INPUT['trans_mode'] = int(INPUT['trans_mode'])
        INPUT['initial_limit'] = int(INPUT['initial_limit'])
        for empty_keys in INPUT.keys():
            if INPUT[empty_keys] == ['']: INPUT[empty_keys] = []
    except:
        print "Error: Invalid input data types!"

    if INPUT['mode'] not in ['GA','ROA', 'Sync']: print "Wrong simulation method! Using 'Sync', 'GA', or 'ROA'"
    return INPUT

def Nodes2Num(booltxt):
    '''Read nodes list from a boolean rules file, output a dictionary, key is node name, value is node index'''
    nodes = []
    for line in booltxt.split('\n'):
        if line != '' and line[0] != '#':
            nodes.extend(GetNodes(line))
    nodes = list(set(nodes))  # remove duplicate and sort
    nodes_sorted = sorted(nodes)
    return (dict(zip(nodes_sorted, range(len(nodes)))))
        
def GetNodes(expression):
    '''convert one line of expression to a node list'''
    nodes = []
    other = ['=', 'and', 'or', 'not']  # remove operator signs
    for node in expression.split():
        node=node.strip('*()')
        if node not in other:
            nodes.append(node.strip('*()'))  # remove * ( ) from the node name
    return nodes

def BooleanFixedNodes(boolfunc, keep={}):
    '''Detect fixed node by boolean function, iteratively until the fixed node number do not increase'''
    new_keep = keep
    while 1:
        keep_num = len(new_keep)
        for line in boolfunc.split('\n'):
            if line != '' and line[0] != '#':
                # try:
                result = IterState(line, new_keep)
                # except:
                #    print 'Error when parsing boolean function:\n%s'%line
                NumVal = len(result[1:])
                SumVal = sum([x[0] for x in result[1:]])
                if SumVal / NumVal in [0, 1]:  # find a fixed node
                    new_keep[result[0][0]] = int(SumVal / NumVal)
        if len(new_keep) == keep_num:  # stop when the number of fixed nodes do not increase
            break
    print 'Identified fixed nodes:'
    for node, value in new_keep.items():
        print '%s\t%s' % (node, value)
    keep.update(new_keep)
    
    return keep
            
def IterState(expression, keep={}):
    '''Iterate all the state of input node and output all the inital state and the value of the target node,
    used to construct truth table.
    Return a list of tuples, the first tuple contain the index of target node and its regulators,
    the rest of the tuple contain all the possible state of the target node and its regulators,
    the first element in the tuple is the state of target'''
    nodes = GetNodes(expression)
    record = []  # to store results
    all_regulator = nodes[1:]  # all regulator of the target
    free_regulator = [node for node in nodes[1:] if node not in keep]  # regulators that free to change
    free_num = len(free_regulator)  # number of free regulators
    target_node = nodes[0]
    record.append(tuple([target_node] + free_regulator))  # record the target node and free regulator
    bool_func = expression.split('=')[1].strip()
    total_ini = 2 ** len(free_regulator)  # n nodes have 2**n combinations
    for node in set(all_regulator) & set(keep.keys()):
        vars()[node] = keep[node]  # set the value of keeped nodes
    for index in xrange(total_ini):
        state = bin(index)[2:].zfill(free_num)  # conver a interger to a boolean string with specified length
        for i in range(len(free_regulator)):
            vars()[free_regulator[i]] = int(state[i])  # set the node variable to logical state, if it is not keeped
        if target_node not in keep:
            target_val = int(eval(bool_func))  # caculate the target node's value, kept nodes are considered implicitly
        else:  # if the node value has been keeped by hand, than used that value iregulate of the state of its regulators
            target_val = int(keep[target_node])
        record.append(tuple([target_val] + [int(n) for n in state]))
    return record

def ConstructTruthTab(boolfunc, keep={}, ini={}):
    '''Construct the truth table that contain all the possibile input state and output state for each node'''
    all_result = []
    all_nodes = set([])  # all nodes in boolean rule file
    target_nodes = set([])  # nodes have regulators in boolean rule file 
    RegNodes = []  # a list contain regulator of each node as a tuple. The tuple index in the list is the target node index
    TruthTab = []  # a list of dictionary contain the truth table for each node. The sequence is in consist with node sequence in mapping
    for line in boolfunc.split('\n'):
        if line != '' and line[0] != '#':
            line_nodes = GetNodes(line)
            target_nodes = target_nodes | set([line_nodes[0]])
            all_nodes = all_nodes | set(line_nodes)
            if line_nodes[0] not in keep.keys():
                all_result.append(IterState(line, keep))
    unmapped = all_nodes - target_nodes - set(keep.keys())  # find the node that do not have regulator, and not specified in the keep list
    for unmapped_id in unmapped:
        all_result.append([(unmapped_id, unmapped_id), (1, 1), (0, 0)])  # if the node do not have any regulate node, then it regulate by itself
    '''insertnode collapsing here'''
    # next step: split the total list into two list, one do not have specified initial state, the other have specified initial state
    new_ini = [node for node in ini.keys() if node not in keep.keys() and node in all_nodes]  # some ini node may be keeped, then remove these nod from ini nodes list 
    new_ini_dic = {}
    for node in new_ini:
        new_ini_dic[node] = int(ini[node])  # generate a dict of new initial state (removed keeped nodes)
    results_ini = []
    free_result=[]
    for node_state in all_result:
        if node_state[0][0] in new_ini:
            results_ini.append(node_state)
        else:
            free_result.append(node_state)
    sorted_free = sorted(free_result, key=lambda x:x[0][0])
    sorted_ini = sorted(results_ini, key=lambda x:x[0][0])
    sorted_all = sorted_free + sorted_ini  # sorted speratly and combine, keep nodes have initial values in the last of the list
    # generate mappings from nodes name to nodes index
    mappings = dict(zip([node[0][0] for node in sorted_all], range(len(sorted_all))))
    
    # generate list of regulators for each node and the truth table, sorted as the mappings
    for each_node in sorted_all:
        state_dic = {}
        regulators = tuple([mappings[node] for node in each_node[0][1:]])
        RegNodes.append(regulators)
        for each_state in each_node[1:]:
            state_dic[each_state[1:]] = each_state[0]
        TruthTab.append(state_dic)
    return RegNodes, TruthTab, mappings, new_ini_dic  # returns 4 objects at the same time or should i just change ini?

def IterOneSync(InitialState):
    '''Iterate model using sychronous method. The most time consuming part, need to be carefully optimized'''
    NewState = [str(TruthTab[i][tuple([int(InitialState[j]) for j in RegNodes[i]])]) for i in range(len(InitialState))]
    '''This strange sentence funtionly equal to (may be faster) :'''
    '''NewState=bitarray([])
    for index in range(len(InitialState)):
        RegState=tuple([InitialState[i] for i in RegNodes[index]]) #get the state of regulators
        NewState.append(TruthTab[index][RegState]) # get the state of targe nodes'''
    
    return ''.join(NewState)

def IterOneROA(InitialState):
    seq=range(len(InitialState))
    shuffle(seq)  # generate a random sequence of updating list
    NewState=list(InitialState)
    for i in seq:
        NewState[i]= str(TruthTab[i][tuple([int(NewState[index]) for index in RegNodes[i]])])
    #NewState = [str(self.TRUTH_TAB[i][tuple([int(NewState[j]) for j in self.REG_NODES[i]])]) for i in seq]
    return ''.join(NewState)

def IterOneGA(InitialState):
    '''Iterate model using asynchronous method (General Asynchronous model: update one random node per step)'''
    update_index=randint(0,len(InitialState)-1)
    NewState=list(InitialState)
    NewState[update_index] = str(TruthTab[update_index][tuple([int(InitialState[index]) for index in RegNodes[update_index]])])
    return ''.join(NewState)
    
def GenIni(NumNodes,limit=0) :
    '''a generator to generate initial state, and omit user defined inital state
    also show the progress at the same time'''
    show_percent = 10  # a variable control the showing frequency of progress
    total_ini = 2 ** NumNodes
    if limit > total_ini:
        limit=total_ini
    if limit == 0:
        abs_step = int((total_ini * show_percent) / 100)
        print 'Total %s (2^%s) initial states' % (2 ** NumNodes, NumNodes)
        index = 0
        while index < total_ini:
            if not index % abs_step: print "{0:.2%} processed".format(index / total_ini)  # show the progress
            IniState = bin(index)[2:].zfill(NumNodes)
            index += 1
            yield IniState
    else:
        abs_step = int((limit * show_percent) / 100)
        print 'Total %s (user specified) initial states' %(limit)
        index = 0
        while index < limit:
            if not index % abs_step: print "{0:.2%} processed".format(index / limit)  # show the progress
            IniState = bin(randint(0,total_ini))[2:].zfill(NumNodes)
            index += 1
            yield IniState
        
def MapStates(Mappings, StateBin):
    '''Map binary state to state index'''
    try:
        return Mappings[StateBin]
    except KeyError:
        index = len(Mappings)
        Mappings[StateBin] = index
        return index
   
    
def Sampling(NumNodes,Rounds=50,Steps=50,PreDefine={},states_limit=0,method='ROA'):
    Collect=defaultdict(int)
    Mappings={}
    if method=='ROA':
        RunMethod=IterOneROA
    else:
        RunMethod=IterOneGA
    FreeNum = NumNodes - len(PreDefine)
    PreState = ''.join([str(PreDefine[i]) for i in sorted(PreDefine)])  # generate a string of predefined state, must be sorted
    for IniState in GenIni(FreeNum,states_limit):
        IniState = ''.join([IniState,PreState]) #append the state of nodes that have specified initial state
        for r in range(Rounds):
            prev=IniState
            prevID=MapStates(Mappings,IniState)
            for s in range(Steps):
                next=RunMethod(prev)
                nextID=MapStates(Mappings,next)
                Collect[(prevID,nextID)] += 1
                prev=next
                prevID=nextID
                
    return Collect,Mappings

def FindSteadyState(NumNodes, PreDefine={},states_limit=0):
    '''The main funtional module, iteratly call IterOneSync to iterate state, and determin whether it is a attractor
    Also time consuming!
    Return a dictionary. Keys are attractors, values are their basins'''
    basin_dic = {}  # store attractor for each state, key as basin state, value as attractor state.
    attractors = {}  # store attractors and its id, may be a little fast using dictionary (hash table)
    att_num = 0  # current id of attractor
    FreeNum = NumNodes - len(PreDefine)
    PreState = ''.join([str(PreDefine[i]) for i in sorted(PreDefine)])  # generate a string of predefined state, must be sorted
    for IniState in GenIni(FreeNum,states_limit):
        IniState = ''.join([IniState,PreState])  # add the initial state to the last of generated ini state
        if IniState in basin_dic:  # omit previously found state
            continue
        state_list = []  # record simulated state
        state_list.append(IniState)
        while 1:  # loop until find the attractor or previously mapped state
            NewState = IterOneSync(state_list[-1])
            if NewState in basin_dic:  # if the new state is previously mapped, than the attractor of the new state must be the same to its father node 
                basin_dic.update(dict(zip(state_list, [basin_dic[NewState]] * len(state_list))))  # set the attractor (id) of this node as the same of its father to 
                break
            else:
                state_list.append(NewState)  # if not previously mapped, append it to the state list
            if state_list[-1] == state_list[-2]:  # if the next state equals the current state, then its is a fixed state in sync model
                attractors[state_list[-1]] = att_num
                basin_dic.update(dict(zip(state_list[:-1], [att_num] * len(state_list[:-1]))))  # add all the states previous to this attractor to the basin of this attractor
                att_num += 1
                break
            elif state_list[-1] in state_list[:-1]:  # if the last step in the list, then these states forms a cyclic steady state
                attractors[tuple(state_list[state_list[:-1].index(state_list[-1]):-1])] = att_num
                basin_dic.update(dict(zip(state_list[:-1], [att_num] * len(state_list[:-1]))))  # add all states in the list to the basin of this attractor
                att_num += 1
                break
    # organizing results
    results = defaultdict(list)
    attractors = dict(zip([value for value in attractors.values()], [key for key in attractors.keys()]))  # interchange values and keys
    for attractor in basin_dic.iteritems():  # another way to interchange value and keys
        results[attractors[attractor[1]]].append(attractor[0])
    return results

   
def FindAttractors(Counts,folder='Data'):
    '''Identifying attractors and basins using NetworkX'''
    print 'Now identifying attractors, please wait...'
    results = {}
    TransNet = nx.DiGraph()
    for source, target in Counts:  # add source and target node to network objects
        TransNet.add_edge(source, target)
    TransNet.remove_edges_from(TransNet.selfloop_edges())
    attractors = nx.attracting_components(TransNet)  # find attractors (SCC with not out edge)
    ReTransNet = TransNet.reverse()  # reverse the directd graph to creat a tree with attactors are the roots
    try:
        os.mkdir(folder)
    except:
        pass
    print 'Now identifying basins for each attractor...'
    for attractor in attractors:
        basin_tree=nx.dfs_tree(ReTransNet,list(attractor)[0])  #just need to find the sons of the first node in attractor
        results[tuple(attractor)]=basin_tree.nodes()
        #results_origin[tuple(attractor)]=[leaf for leaf in basin if ReTransNet.out_degree(leaf) == 0] # record initial states of attractors
        #AttNet=TransNet.subgraph(attractor)
        #nx.write_edgelist(AttNet,'%s/Attractor%s.txt'%(folder,attractors.index(attractor)),data=False)
    print 'Writing out transition graph in %s/TransGraph.txt'%folder
    nx.write_edgelist(TransNet,'%s/TransGraph.txt'%folder,data=False)
    return results
        
def BasinIntersection(Attractors):
    import itertools
    inter_result={}
    exclude_result={}
    for combine_n in range(2,len(Attractors)+1):
        Att_combine=itertools.combinations(Attractors.keys(),combine_n)
        for each_combine in Att_combine:
            exclude_result[each_combine]=[] #to store a list of excluded basin of each attractor under certain combination
            basin_set=[set(Attractors[x]) for x in each_combine]
            intersection=set.intersection(*basin_set)
            if len(intersection) != 0:
                inter_result[each_combine]=intersection
            for each_att in Attractors:
                exclude_result[each_combine].append((each_att,set(Attractors[each_att])-intersection))
    return inter_result,exclude_result

def EstimateSimple(Counts):

    Populations = np.array(Counts.sum(1).flatten())[0]

    return Populations
        

def SummaryEnergy(StatesMapping, NodesMapping, Energy='Data/Energy.txt', Outdir='Data'):
    '''Now summarizing energy for each state...'''
    States_sum = {}
    nodes_list = sorted(NodesMapping, key=lambda x: NodesMapping[x])
    index = 0
    for line in open(Energy):
        States_sum[index] = float(line.split('\t')[1].strip())
        index += 1 
    Sorted_sum = sorted(States_sum.iteritems(), key=lambda x: x[1])
    Output = file('Data/Summary.csv', 'w')
    Output.writelines('%s\n' % (','.join(['Index', 'Energy'] + nodes_list)))
    for each_state in Sorted_sum:
        Output.writelines('%s,%4.3f,%s\n' % (each_state[0], each_state[1], ','.join(list(StatesMapping[each_state[0]]))))
    Output.close()
    
def SummarySync(Attractors, keep, NodesMapping, Outdir='Data'):
    '''Summary result of attractors and basins (Sync mode)'''
    print 'Now summarizing results...'
    p_att = {}  # record point attractor
    c_att = {}  # record cyclic attractor
    at_array = []
    fixed_nodes = []  # record nodes that do not change in all steady state
    Nodes = sorted(NodesMapping, key=lambda x:NodesMapping[x])
    try:
        os.mkdir(Outdir)
    except:
        pass
    point_att = file('%s/Point_attractors.csv' % Outdir, 'w')
    cyclic_att = file('%s/Cyclic_attractors.csv' % Outdir, 'w')
    all_att = file('%s/Attractors.bin' % Outdir, 'wb')
    point_att.writelines('%s\n' % (','.join(['Attractor', 'Basin Num'] + Nodes)))
    cyclic_att.writelines('%s\n' % (','.join(['Attractor', 'Basin Num'] + Nodes)))
    for state in Attractors.keys():
        if type(state) is tuple:   #a cyclic attractor
            c_att[state] = len(Attractors[state])
            for each_state in state:
                at_array.append([int(node) for node in each_state])
                cyclic_att.writelines("'%s',%s,%s\n" % (each_state, c_att[state], ','.join(list(each_state))))
            cyclic_att.writelines('\n')
        else:   # a point attractor
            at_array.append([int(node) for node in state])
            p_att[state] = len(Attractors[state])
            point_att.writelines("'%s',%s,%s\n" % (state, p_att[state], ','.join(list(state))))
    
    point_att.close()
    cyclic_att.close()
    cPickle.dump(Attractors, all_att, 2)
    all_att.close()
    print '\nNumber of point attractors: %s' % len(p_att.keys())
    for state in p_att.keys():
        print 'Attractor:%s\tNumber of basins:%s' % (state, p_att[state])
    print '\nNumber of cyclic attractors: %s' % len(c_att.keys())
    for state in c_att.keys():
        print 'Attractor:%s\tNumber of basins:%s' % (state, c_att[state])
    at_array = np.array(at_array).transpose()

    # find nodes that do not change in all attractors (nodes whose state do not relate to the steady state and can be removed from the boolean function)
    for i in range(len(at_array)):
        state_avg = at_array[i].sum() / len(at_array[i])
        if state_avg == 0 or state_avg == 1:
            fixed_nodes.append((Nodes[i], int(state_avg)))
    fixed_nodes.extend(keep.items())
    fixed_out = file('%s/Fixed_nodes.txt' % Outdir, 'w')
    print '\nNodes that do not change in all the steady states:'
    for node, state in fixed_nodes:
        print '%s\t%s' % (node, state)
        fixed_out.writelines('%s\t%s\n' % (node, state))
    print '\nAttractors information have written in %s/Point_attractors.csv and %s/Cyclic_attractors.csv file' % (Outdir, Outdir)
    print '\nFixed nodes list have been saved to %s/Fixed_nodes.txt' % Outdir
    fixed_out.close()            
    
def SummaryAsync(Attractors, keep, NodesMapping, StatesMapping,Outdir='Data'):
    '''Summary result of attractors and basins (results from asynchronous mode)'''
    
    def StringAdd(xlist,ystring):
            return [x+int(y) for x,y in zip(xlist,ystring)]
        
    def AverageStates(states_list):
        total_states=len(states_list)
        sum_states=[0]*len(Nodes)
        for each_state in states_list:
            sum_states=StringAdd(sum_states,state_dic[each_state])
        avg_states=['%2.2f'%(x/total_states) for x in sum_states]
        return avg_states
        
    print 'Now summarizing results...'
    state_dic = dict(zip(StatesMapping.values(), StatesMapping.keys())) #reverse the mapping
    p_att = {}  # record point attractor
    c_att = {}  # record cyclic attractor
    Nodes = sorted(NodesMapping, key=lambda x:NodesMapping[x])
    point_att = file('%s/Point_attractors.csv' % Outdir, 'w')
    cyclic_att = file('%s/Cyclic_attractors.csv' % Outdir, 'w')
    all_att = file('%s/Attractors.bin' % Outdir, 'wb')
    point_att.writelines('%s\n' % (','.join(['Attractor', 'Basin Num'] + Nodes)))
    cyclic_att.writelines('%s\n' % (','.join(['Attractor', 'Basin Num'] + Nodes)))
    for state in Attractors.keys():
        if len(state) > 1:   #a cyclic attractor or complex attractor
            c_att[state] = len(Attractors[state])
            for each_state in state:
                cyclic_att.writelines("%s,%s,%s\n" % (each_state, c_att[state], ','.join(list(state_dic[each_state]))))
            cyclic_att.writelines('\n')
        else:   # a point attractor
            p_att[state] = len(Attractors[state])
            point_att.writelines("%s,%s,%s\n" % (state[0], p_att[state], ','.join(list(state_dic[state[0]]))))
    
    point_att.close()
    cyclic_att.close()
    cPickle.dump(Attractors, all_att, 2)
    all_att.close()
    print '\nNumber of point attractors: %s' % len(p_att.keys())
    for state in p_att.keys():
        print 'Attractor:%s\tNumber of basins:%s' % (state, p_att[state])
    print '\nNumber of cyclic attractors: %s' % len(c_att.keys())
    for state in c_att.keys():
        print 'Attractor:%s\tNumber of basins:%s' % (state, c_att[state])
    fixed_out = file('%s/Fixed_nodes.txt' % Outdir, 'w')
    print '\nNodes that do not change:'
    for node, state in keep.items():
        print '%s\t%s' % (node, state)
        fixed_out.writelines('%s\t%s\n' % (node, state))
    print '\nAttractors information have written in %s/Point_attractors.csv and %s/Cyclic_attractors.csv file' % (Outdir, Outdir)
    print '\nFixed nodes list have been saved to %s/Fixed_nodes.txt' % Outdir
    fixed_out.close()

    # Summary initial informations for each attractors
    print 'Now summarizing  basins for each attractor'
    SumOut=file('%s/Summary_basin.txt'%Outdir,'w')
    SumOut.writelines('Summary of basin states information for each attractor\n')
    for each_att in Attractors:
        SumOut.writelines('Attractor: (%s)\n'%(','.join(map(str,list(each_att)))))
        SumOut.writelines('Number of basins: %s\n'%len(Attractors[each_att]))
        SumOut.writelines('Summary for basin states of each attractor:\n')
        initial_avg=AverageStates(Attractors[each_att])
        nodes_state=zip(Nodes,initial_avg)
        for node,state in nodes_state:
            SumOut.writelines('%s\t%s\n'%(node,state))
        SumOut.writelines('\n\n')

    #Summary for interactions of initial state for each attractors
    print 'Now summarizing basin intersections for each attractor'
    SumOut.writelines('Summary of initial state intersection for each attractors\n')
    inter,exclude=BasinIntersection(Attractors)
    for each_combine in inter:
        for att in each_combine:
            SumOut.writelines('Attractor: (%s)\n'%(','.join(map(str,list(att)))))
        SumOut.writelines('Number of overlap basins: %s\n'%len(inter[each_combine]))
        initial_avg=AverageStates(inter[each_combine])
        nodes_state=zip(Nodes,initial_avg)
        for node,state in nodes_state:
            SumOut.writelines('%s\t%s\n'%(node,state))
        SumOut.writelines('\n')
        for each_att in exclude[each_combine]:
            SumOut.writelines('Number of exclude basin for:\n')
            SumOut.writelines('Attractor: (%s)\n'%(','.join(map(str,list(each_att[0])))))
            exclude_avg=AverageStates(each_att[1])
            nodes_state=zip(Nodes,exclude_avg)
            for node,state in nodes_state:
                SumOut.writelines('%s\t%s\n'%(node,state))
            SumOut.writelines('\n')

    SumOut.close()
    
    return
    

if __name__ == '__main__':
    if sys.argv[1]:
        para_file=sys.argv[1]
    else:
        para_file = 'steady.in'
    INPUT = ParaParser(para_file)
    text = open(INPUT['rules']).read()
    ini_state = {}  # define inital states of each nodes,initial state include turn_on or turn_off nodes
    for on_nodes in INPUT['ini_on'] + INPUT['turn_on']:
        ini_state[on_nodes] = 1
    for off_nodes in INPUT['ini_off'] + INPUT['turn_off']:
        ini_state[off_nodes] = 0
    keep_state = {}  # state of the turn on and turn off nodes,used to construct truth table
    for on_nodes in INPUT['turn_on']:
        keep_state[on_nodes] = 1
    for off_nodes in INPUT['turn_off']:
        keep_state[off_nodes] = 0
    keep_new = BooleanFixedNodes(text, keep_state)
    RegNodes, TruthTab, nodes_map, ini_new = ConstructTruthTab(text, keep_new, ini_state)  # after truth table construction the nodes mapping  and initial nodes have changed.
    FreeNodes = len(RegNodes)
    print 'Now identifying attractors, using %s method, please wait...' % INPUT['mode']
    #####two modes are available: sync and async
    if INPUT['mode'] == 'sync':
        all_attractors = FindSteadyState(FreeNodes, ini_new, INPUT['initial_limit'])
        SummarySync(all_attractors, keep_new, nodes_map)
    else:
        count_m, state_map = Sampling(FreeNodes,INPUT['rounds'],INPUT['steps'],ini_new,INPUT['initial_limit'],method=INPUT['mode'])
        all_attractors=FindAttractors(count_m)
        SummaryAsync(all_attractors,keep_new,nodes_map,state_map)

