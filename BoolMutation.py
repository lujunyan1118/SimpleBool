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
import random
import sys

class Model:
    def __init__(self,INPUT,mut_nodes=[]):
        random.seed()    
        self.KEEP={}
        self.INITIAL={}
        self.REG_NODES=[]
        self.TRUTH_TAB=[]
        self.MAPPING={}
        self.INPUT={}
        self.FINAL={}
        
        def GetNodes(expression):
            '''convert one line of expression to a node list'''
            nodes = []
            other = ['=', 'and', 'or', 'not']  # remove operator signs
            for node in expression.split():
                node=node.strip('*() ')
                if node not in other:
                    nodes.append(node)  # remove * ( ) from the node name
            return nodes
        
        def IterState(expression, keep):
            '''Iterate all the state of input node and output all the inital state and the value of the target node,
            used to construct truth table.
            Return a list of tuples, the first tuple contain the index of target node and its regulators,
            the rest of the tuple contain all the possible state of the target node and its regulators,
            the first element in the tuple is the state of target'''
            nodes = GetNodes(expression)
            record = []  # to store results
            all_regulator = nodes[1:]  # all regulator of the target
            target_node = nodes[0]
            record.append(tuple([target_node] + all_regulator))  # record the target node and free regulator
            bool_func = expression.split('=')[1].strip()
            total_ini = 2 ** len(all_regulator)  # n nodes have 2**n combinations
            for node in set(all_regulator) & set(keep.keys()):
                vars()[node] = keep[node]  # set the value of keeped nodes
            for index in xrange(total_ini):
                state = bin(index)[2:].zfill(len(all_regulator))  # conver a interger to a boolean string with specified length
                for i in range(len(all_regulator)):
                    vars()[all_regulator[i]] = int(state[i])  # set the node variable to logical state, if it is not keeped
                if target_node not in keep:
                    target_val = int(eval(bool_func))  # caculate the target node's value, kept nodes are considered implicitly
                else:  # if the node value has been keeped by hand, than used that value iregulate of the state of its regulators
                    target_val = int(keep[target_node])
                record.append(tuple([target_val] + [int(n) for n in state]))
            return record

        def ConstructTruthTab(booltext, keep):
            '''Construct the truth table that contain all the possibile input state and output state for each node'''
            all_result = []
            all_nodes = set([])  # all nodes in boolean rule file
            target_nodes = set([])  # nodes have regulators in boolean rule file 
            RegNodes = []  # a list contain regulator of each node as a tuple. The tuple index in the list is the target node index
            TruthTab = []  # a list of dictionary contain the truth table for each node. The sequence is in consist with node sequence in mapping
            for line in booltext.split('\n'):
                if line.strip() != '' and line[0] != '#':
                    line_nodes = GetNodes(line)
                    target_nodes = target_nodes | set([line_nodes[0]])
                    all_nodes = all_nodes | set(line_nodes)
                    if line_nodes[0] not in keep.keys():
                        try:
                            all_result.append(IterState(line, keep))
                        except:
                            print "Expressing error of boolean function"
                            print line
                    else:             #if the node has been kept
                        all_result.append([(line_nodes[0], line_nodes[0]), (1, 1), (0, 0)])
        
            unmapped = all_nodes - target_nodes  # find the node that do not have regulator, and not specified in the keep list
            for unmapped_id in unmapped:
                all_result.append([(unmapped_id, unmapped_id), (1, 1), (0, 0)])  # if the node do not have any regulate node, then it regulate by itself
            
            sorted_all = sorted(all_result, key=lambda x:x[0][0]) 
            mappings = dict(zip([node[0][0] for node in sorted_all], range(len(sorted_all))))
            # generate list of regulators for each node and the truth table, sorted as the mappings
            for each_node in sorted_all:
                state_dic = {}
                regulators = tuple([mappings[node] for node in each_node[0][1:]])
                RegNodes.append(regulators)
                for each_state in each_node[1:]:
                    state_dic[each_state[1:]] = each_state[0]
                TruthTab.append(state_dic)
            return RegNodes, TruthTab, mappings #
    
        self.INPUT=INPUT
        
        for on_nodes in INPUT['ini_on']:
            self.INITIAL[on_nodes] = True
            
        for off_nodes in INPUT['ini_off']:
            self.INITIAL[off_nodes] = False

        for on_nodes in INPUT['turn_on']:
            self.INITIAL[on_nodes] = True
            self.KEEP[on_nodes] = True
        for off_nodes in INPUT['turn_off']:
            self.KEEP[off_nodes] = False
            self.INITIAL[off_nodes] = False
                    
        for node,state in mut_nodes:
            self.KEEP[node]=state
            self.INITIAL[node]=state
        
        self.REG_NODES,self.TRUTH_TAB,self.MAPPING=ConstructTruthTab(open(INPUT['rules']).read(),self.KEEP)
        model_verbose={'Sync':'Synchronous','GA':'General Asynchrounous','ROA':'Random Order Asynchrounous'}
        print '''Model initialization completed!
        
Total nodes number:    %s
Simulation steps:    %s
Simulation rounds:    %s
Simulation mode:    %s
        '''%(len(self.MAPPING.keys()),INPUT['steps'],INPUT['rounds'],model_verbose[INPUT['mode']])

    def GetNodes(self):
        return sorted(self.MAPPING)
    
    def GetFixed(self,file_out='steady.txt'):
        all_nodes=self.GetNodes()
        on_nodes=[]
        off_nodes=[]
        output=file(file_out,'w')
        for node in all_nodes:
            output.writelines('%s\t%s\n'%(node,self.FINAL[node]))
            if self.FINAL[node] == 0:
                off_nodes.append(node)
            elif self.FINAL[node] == 1:
                on_nodes.append(node)
        print '''%s nodes stabilized on 'ON' state: %s '''%(len(on_nodes),','.join(on_nodes))
        print '''%s nodes stabilized on 'OFF' state: %s '''%(len(off_nodes),','.join(off_nodes)) 
        output.close()


    def IterModel(self,missing='random',InitialList=[]):
        
        final_all=[]
        traj_all=[]
        steps=self.INPUT['steps']
        rounds=self.INPUT['rounds']
        missing=self.INPUT['missing']
        collect=[[0]*len(self.MAPPING)]*(steps+1)
        
        def IterOneSync(InitialState):
            '''Iterate model using sychronous method. The most time consuming part, need to be carefully optimized'''
            NewState = [str(self.TRUTH_TAB[i][tuple([int(InitialState[j]) for j in self.REG_NODES[i]])]) for i in range(len(InitialState))]
            return ''.join(NewState)
        
        def IterOneAsync(InitialState):
            '''Iterate model using asynchronous method (General Asynchronous model: update one random node per step)'''
            update_index=random.randint(0,len(InitialState)-1)
            NewState=list(InitialState)
            NewState[update_index] = str(self.TRUTH_TAB[update_index][tuple([int(InitialState[index]) for index in self.REG_NODES[update_index]])])
            return ''.join(NewState)
        
        def IterOneROA(InitialState):
            seq=range(len(InitialState))
            random.shuffle(seq)  # generate a random sequence of updating list
            NewState=list(InitialState)
            for i in seq:
                NewState[i]= str(self.TRUTH_TAB[i][tuple([int(NewState[index]) for index in self.REG_NODES[i]])])
            #NewState = [str(self.TRUTH_TAB[i][tuple([int(NewState[j]) for j in self.REG_NODES[i]])]) for i in seq]
            return ''.join(NewState)
                
            
        def GenInitial():
            initial_state=[]
            for node in sorted(self.MAPPING.keys()):
                if node in self.INITIAL:
                    initial_state.append(str(int(self.INITIAL[node])))
                else:
                    if missing=='random':
                        initial_state.append(random.choice(['0','1']))
                    else:
                        initial_state.append(str(int(missing)))
            return ''.join(initial_state)
        
        def StringAdd(xlist,ystring):
            return [x+int(y) for x,y in zip(xlist,ystring)]
        
        def divide(x):
            return x/rounds
                    
        for r in range(rounds):
            traj=[]
            if InitialList==[]:
                ini_state=GenInitial()
            else:
                ini_state=InitialList[r]
            traj.append(ini_state)
            prev=ini_state
            collect[0]=StringAdd(collect[0],prev)
            for s in range(steps):
                if self.INPUT['mode']=='Sync':
                    next_state=IterOneSync(prev)
                elif self.INPUT['mode']=='GA':
                    next_state=IterOneAsync(prev)
                elif self.INPUT['mode'] == 'ROA':
                    next_state=IterOneROA(prev)
                traj.append(next_state)
                collect[s+1]=StringAdd(collect[s+1],next_state)
                prev=next_state
            traj_all.append(traj)
            final_all.append(traj[-1])
        
        out={}
        normalized=[map(divide,each_step) for each_step in collect]
        
        nodes_list=self.GetNodes()
        for node_i in range(len(nodes_list)):
            out[nodes_list[node_i]]=[state[node_i] for state in normalized]
            self.FINAL[nodes_list[node_i]]=out[nodes_list[node_i]][-1]
        
            
        
        return out,final_all

def ParaParser(ParaFile): 
    '''#parser parameters for simulation and transition matrix building'''
    INPUT = {'rules'    :    'rules.txt',
             'ini_on'   :    '',
             'ini_off'  :    '',
             'turn_on'  :    '',
             'turn_off' :    '',
             'rounds'   :    1,
             'steps'    :    1,
             'equi_steps':  0,
             'mode'     :    'ROA',
             'mutation_list' :  'nodes.txt',
             'mutation_mode'    :   'single',
             'observe_list' :   '',
             'keep_state'   :   'False',
             'missing'  :   'random'
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
        INPUT['rules'] = str(INPUT['rules'].strip())
        INPUT['ini_on'] = [node.strip() for node in INPUT['ini_on'].split(',')]
        INPUT['ini_off'] = [node.strip() for node in INPUT['ini_off'].split(',')]
        INPUT['turn_on'] = [node.strip() for node in INPUT['turn_on'].split(',')]
        INPUT['turn_off'] = [node.strip() for node in INPUT['turn_off'].split(',')]
        INPUT['observe_list'] = [node.strip() for node in INPUT['observe_list'].split(',')]
        INPUT['mutation_list'] = INPUT['mutation_list'].split(',')
        INPUT['keep_state'] = INPUT['keep_state'].split(',')
        INPUT['rounds'] = int(INPUT['rounds'])
        INPUT['steps'] = int(INPUT['steps'])
        INPUT['mode'] = str(INPUT['mode'])
        INPUT['equi_steps']=int(INPUT['equi_steps'])
        INPUT['mutation_mode']=str(INPUT['mutation_mode'])
        for empty_keys in INPUT.keys():
            if INPUT[empty_keys] == ['']: INPUT[empty_keys] = []
    except:
        print "Error: Invalid input data types!"

    if INPUT['mode'] not in ['GA', 'Sync','ROA']: print "Wrong simulation method! Using 'GA', 'ROA' or 'Sync'"
    return INPUT


def ChangeInitial(NodesList,States,MutePairs):
    OutStates=[]
    for EachState in States:
        State=list(EachState)
        for MuteNode,MuteState in MutePairs:
            State[NodesList.index(MuteNode)]=str(int(MuteState))
        OutStates.append(''.join(State))
    return OutStates

def simu_mutation(INPUT,missing='random'):
    '''Do simulated mutation expriment of the model'''
    
    
    mut_result={} # to store the mutated result
    #generate mutation list, only single mutation now
    mut_list=GenerateCombinations(INPUT)

    if INPUT['equi_steps'] == 0:
        for mute_pair in mut_list:
            mut_result[mute_pair]=[]
            print 'Now doing mutation',mute_pair
            model=Model(INPUT,mute_pair)
            result,final_all=model.IterModel(missing=missing)
            for out_node in INPUT['observe_list']:
                mut_result[mute_pair].append('%2.2f'%(sum(result[out_node][-50:])/50))     
    
    elif INPUT['equi_steps'] > 0:
        print 'Now running pre-mutation simulation to reach steady state\n'
        model=Model(INPUT)
        AllNodes=model.GetNodes()
        result_pre,final_pre=model.IterModel(missing) #pre-equilibration run
        print 'Now running mutation scanning\n'
        for mute_pair in mut_list:
            mut_result[mute_pair]=[]
            print 'Now doing mutation',mute_pair
            new_model=Model(INPUT,mute_pair)
            pre_modified=ChangeInitial(AllNodes,final_pre,mute_pair)
            result,final=new_model.IterModel(missing=missing, InitialList=pre_modified) #actual mutation run for each mutation
            for out_node in INPUT['observe_list']:
                mut_result[mute_pair].append(('%2.2f'%(sum(result_pre[out_node][-50:])/50),'%2.2f'%(sum(result[out_node][-50:])/50)))

    print "Write mutation result to 'mutation_result.csv' "
    output=open('mutation_result.csv','w')
    if INPUT['equi_steps'] == 0: #direct mutation
        output.writelines('MutNodes,MutStates,%s\n'%(','.join(INPUT['observe_list'])))
        for mute_pair in mut_list:
            nodes=[]
            states=[]
            for node,state in mute_pair:
                nodes.append(node)
                states.append(str(state))
            output.writelines('%s,%s,%s\n'%('/'.join(nodes),'/'.join(states),','.join(mut_result[mute_pair])))
            
    elif INPUT['equi_steps'] > 0:  #mutation with pre-equilibration steps
        out_list=[]
        for node in INPUT['observe_list']:
            out_list.append(node)
            out_list.append(node+'(mut)')
        output.writelines('MutNodes,MutStates,%s\n'%(','.join(out_list)))
        for mute_pair in mut_list:
            nodes=[]
            states=[]
            
            for node,state in mute_pair:
                nodes.append(node)
                states.append(str(state))
            
            result_list=[each_result for each_pair in mut_result[mute_pair] for each_result in each_pair] # collapse results
            output.writelines('%s,%s,%s\n'%('/'.join(nodes),'/'.join(states),','.join(result_list)))
    output.close()

def GenerateCombinations(INPUT):
    StrBool={'True':True,'False':False}
    NodeState=[]
    Combine=[]
    
    for node_list in INPUT['mutation_list']:
        for node in open(node_list):
            NodeState.append((node.strip(),StrBool[INPUT['keep_state'][INPUT['mutation_list'].index(node_list)]]))
        
    if INPUT['mutation_mode']=='single':
        for mute_state in NodeState:
            Combine.append((mute_state,))
        return Combine
    elif INPUT['mutation_mode']=='double':
        for i in range(len(NodeState)):
            for j in range(i+1,len(NodeState)):
                if NodeState[i][0] != NodeState[j][0]:
                    Combine.append((NodeState[i],NodeState[j]))
        print 'Total %s combinations.'%(len(Combine))
        return Combine
                
if __name__ == '__main__':
    try:
        para=ParaParser(sys.argv[1])
    except:
        para=ParaParser('mutation.in')
    simu_mutation(para)
