from transform.opaggregate import OperAggregate
from transform.typeanalysis import TypeAnalysis
from transform.branchdivergence import BranchDivergence
from transform.xmad_to_imad import XmadToImad

class Transforms:
    def __init__(self, name):
        self.name = name
        self.passes = []

        # Add passes
        # # Add operator aggregation pass
        # self.passes.append(OperAggregate("operator aggregation"))
        # Add type analysis pass
        self.passes.append(TypeAnalysis("type analysis"))
        # Convert XMAD to IMAD pass
        self.passes.append(XmadToImad("xmad to imad"))
        # Add branch divergence detection pass
        self.passes.append(BranchDivergence())

    def apply(self, module):
        for tranpass in self.passes:
            tranpass.apply(module)
