class Edge():
    def __init__(self,start,end,**kwargs):
        self.start = start
        self.end = end
        self.__dict__.update(kwargs)
    def __str__(self):
        s= 'start: '+str(self.start)+ " end: "+str(self.end)+ " "
        if 'time' in self.__dict__:
            s += "time: "+str(self.__dict__['time'])
        return s
    
class Node():
    def __init__(self,id,**kwargs):
        self.id = id
        self.__dict__.update(kwargs)
     
def prepare_sample_probs(edge):
    if len(edge.outgoing_edges) == 0:
        return []
    out_degree = len(edge.outgoing_edges)
    nbr_sample_probs = [1.0 / float(out_degree)] * out_degree
    return nbr_sample_probs