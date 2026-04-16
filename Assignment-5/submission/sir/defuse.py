from sir.instruction import Instruction

class DefUse:
    def __init__(self):
        self.Def2Uses = {}
        self.Use2Def = {}
            
    # Add def-use chaiin
    def AddDefUse(self, DefInst, UseInst):
        self.Def2Uses[DefInst].append(UseInst)
        self.Use2Def[UseInst] = DefInst

    # Get uses from the given def
    def GetUses(self, DefInst):
        return self.Def2Uses[DefInst]

    # Get def from the given use
    def GetDef(self, UseInst):
        return self.Use2Def[UseInst]
