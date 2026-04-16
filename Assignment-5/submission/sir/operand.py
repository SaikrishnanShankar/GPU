from lift.lifter import Lifter

REG_PREFIX = 'R'
ARG_PREFIX = 'c[0x0]'
ARG_OFFSET = 320 # 0x140
THREAD_IDX = 'SR_TID'

class InvalidOperandException(Exception):
    pass

class Operand:
    def __init__(self, Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsDim, IsThreadIdx):
        self._Name = Name
        self._Suffix = Suffix
        self._ArgOffset = ArgOffset
        self._IsReg = IsReg
        self._IsArg = IsArg
        self._IsDim = IsDim
        self._IsThreadIdx = IsThreadIdx
        self._Skipped = False
        
        self._TypeDesc = "NOTYPE"
        self._IRType = None
        self._IRRegName = None

        if Reg and '+' in Reg:
            self._Reg = Reg.split('+')[0]
            self._MemAddrOffset = Reg.split('+')[1]
        else:
            self._Reg = Reg
            self._MemAddrOffset = None

    @property
    def Name(self):
        return self._Name
    
    @property
    def Reg(self):
        return self._Reg

    @property
    def IsReg(self):
        return self._IsReg
    
    @property
    def IsArg(self):
        return self._IsArg

    @property
    def IsThreadIdx(self):
        return self._IsThreadIdx
    
    @property
    def ArgOffset(self):
        return self._ArgOffset

    @property
    def TypeDesc(self):
        return self._TypeDesc

    @property
    def Skipped(self):
        return self._Skipped
    
    # Set the type description for operand
    def SetTypeDesc(self, Ty):
        self._TypeDesc = Ty

    # Set the skip flag
    def SetSkip(self):
        self._Skipped = True
        
    # Get the type description
    def GetTypeDesc(self):
        return self._TypeDesc

    def HasTypeDesc(self):
        return self._TypeDesc != "NOTYPE"

    def GetIRType(self, lifter):
        if self._IRType == None:
            self._IRType = lifter.GetIRType(self._TypeDesc)

        return self._IRType

    def GetIRRegName(self, lifter):
        if self._IRRegName == None:
            self._IRRegName = self._Reg + self._TypeDesc

        return self._IRRegName
    
    def dump(self):
        print("operand: ", self._Name, self._Reg)
    
