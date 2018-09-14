from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import json
import pickle
import os

def list():
    '''
    Returns all models
    '''
    model_list = os.listdir('./examples/models')

    for i, x in enumerate(model_list):
        model_list[i] = x.split('.')[0]

    return json.dumps({'list': model_list})


def model_present(model_name):
    '''
    checks a model present or not
    '''

    model_list = os.listdir('./examples/models')

    for i, x in enumerate(model_list):
        model_list[i] = x.split('.')[0]

    if model_name in model_list:
        return True
    else:
        return False


def describe(model_name):
    '''
    '''

    if not model_present(model_name):
        return "Model not found"

    # getting model pickle
    model_pickle = "./examples/models/"+model_name+".pickle"
    pickle_in = open(model_pickle, "rb")
    model = pickle.load(pickle_in)

    # getting model mapper
    model_json_file = "./examples/data/"+model_name+".json"
    with open(model_json_file) as f:
        model_json = json.load(f)


    root = []
    for x in model.in_degree:
        if x[1] == 0:
            root.append(x[0])
    leaf = []
    for x in model.out_degree:
        if x[1] == 0:
            leaf.append(x[0])

    for i in root:
        for j in leaf:
            #to remove leaf nodes for which some of the root nodes have no influence
            if str(model.is_active_trail(i,j))=='False':
                leaf.remove(j)
    Root=[]
    Leaf=[]
    m = model_json["nodes"]
    for r in root:
        for r1 in m:
            if r==r1:
                Root.append(m[r])
    for l in leaf:
        for l1 in m:
            if l==l1:
                Leaf.append(m[l])
    if len(Leaf)>1:
        op="the Leaves"
    else:
        op="the  Leaf"
    if len(Root)>1:
        op1="the Roots"
    else:
        op1="the Root"


    s=""
    last=" states,"
    rlen=len(root)
    for r,r1 in zip(root,Root):

        length=model.get_cardinality(r)
        if rlen!=1:
            if rlen==2:
                last=" states"
            s+=str(r1)+" has "+ str(length) +last
        else:
            s+=first+str(r1)+" has "+ str(length) +last
        rlen=rlen-1
        if rlen==1:
            first=" and "
            last=" states."

    s2=""
    last2=" states,"
    leaflen=len(leaf)
    for l,l1 in zip(leaf,Leaf):

        length2=model.get_cardinality(l)
        if leaflen==1:
            last2=" states"
            first=""
        if leaflen!=1:
            if leaflen==2:
                last2=" states"
            s2+=str(l1)+" has "+ str(length2) +last2
        else:
            s2+=first+str(l1)+" has "+ str(length2) +last2
        leaflen=leaflen-1
        if leaflen==1:
            first=" and "
            last=" states."


    return ("The "+model_name +" model "  " has "+str(Root).strip('[]')+" as " +op1+ " and "+str(Leaf).strip('[]')+" as "+op+". " +s+s2)

def infer(model_name,output_node,observe):
    '''description: query the given bayesian model
    Parameters
        ----------
        model: pgmpy Bayesian Object
        returns: result of the given query
    '''
    class SimpleInference(Inference):
        ''' custom inference'''
        def query(self, var, evidence):
            # self.factors is a dict of the form of {node: [factors_involving_node]}
            factors_list = set(itertools.chain(*self.factors.values()))
            product = factor_product(*factors_list)
            reduced_prod = product.reduce(evidence, inplace=False)
            reduced_prod.normalize()
            var_to_marg = set(self.model.nodes()) - set(var) - set([state[0] for state in evidence])
            marg_prod = reduced_prod.marginalize(var_to_marg, inplace=False)
            return marg_prod

    if not model_present(model_name):
        return "Model not found"

    # getting model pickle
    model_pickle = "./examples/models/"+model_name+".pickle"
    pickle_in = open(model_pickle, "rb")
    model = pickle.load(pickle_in)

    # getting model json
    model_json_file = "./examples/data/"+model_name+".json"
    with open(model_json_file) as f:
        model_json = json.load(f)

    m = model_json["nodes"]

    #for infer
    for key in observe:
        for node in m:
            if key==m[node]:
                observe[node]=observe[key]
                del observe[key]
    #getting mapping for output node
    for node in m:
        if m[node]==output_node:
            outp_nde=node

    # giving evidence (array of tuples)
    evidence_array =observe.items()
    # infer object
    infer = SimpleInference(model)
    # working for only one evidence
    result = infer.query(var=outp_nde, evidence=evidence_array).values[1]

    output_value={'outp_v':str(result)}

    return (json.dumps(output_value))
