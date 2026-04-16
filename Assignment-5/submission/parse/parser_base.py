from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.defuse import DefUse
from sir.operand import Operand
from sir.operand import InvalidOperandException

class NoParsingEffort(Exception):
    pass

class UnmatchedControlCode(Exception):
    pass

REG_PREFIX = 'R'
ARG_PREFIX = 'c[0x0]'
ARG_OFFSET = 320 # 0x140
THREAD_IDX = 'SR_TID'

PTRREG_PREFIX = '[R'
PATH_PREFIX = 'P'
PTR_PREFIX = '['
PTR_SUFFIX = ']'

class SaSSParserBase:
    def __init__(self, isa, file):
        self.file = file

    # Parse the SaSS text file
    def apply(self):
        # List of functions
        Funcs = []

        # The current function that is under parsing
        CurrFunc = None

        # The instruction list
        Insts = []

        # Remove curly braces from dual issue
        lines = self.file.split('\n')
        modified_lines = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if '{' in line:
                line = line.replace("{", "")
            if "}" in line:
                if line[-1] != "/":
                    line = line + lines[i + 1]
                    line = line.replace("\n", "")
                    skip_next = True
                line = line.replace("}",";")
            modified_lines.append(line)

        lines = modified_lines

        # Main loop that parse the SaSS text file
        for line_num, line in enumerate(lines):
            # Process lines in SaSS text file

            # Process function title and misc
            PrevFunc = CurrFunc
            CurrFunc = self.CreateFunction(line, PrevFunc, Insts)
            if PrevFunc != CurrFunc and CurrFunc != None:
                # Just create a new functino
                if PrevFunc != None:
                    # Add the parsed function
                    Funcs.append(PrevFunc)
                    # Clean the instruction list
                    Insts = []
                    
                continue
            else:
                CurrFunc = PrevFunc
                
            # Process function body
            self.ParseFuncBody(line, Insts, CurrFunc)

        # Process the last function
        if CurrFunc != None:
            # Create def-use chain
            CurrFunc.DefUse = self.CreateDefUse(Insts)
            
            # Wrap up previous function by creating control-flow graph
            CurrFunc.blocks = self.CreateCFG(self.SplitBlocks(Insts))

            # Add the parsed function
            Funcs.append(CurrFunc)
            
        return Funcs

    # Parse the file line to create new function
    def CreateFunction(self, line, PrevFunc, Insts):
        if (not ("/*" in line and "*/" in line)):
            # Check function start
            if ("Function : " in line):
                # Wrap up previous function by creating control-flow graph
                if PrevFunc != None:
                    CurrFunc.blocks = self.CreateCFG(self.SplitBlocks(Insts))
                    
                # Check the function name
                items = line.split(' : ')

                # Create new function
                return Function(items[1])

        return None

    # Parse the function body from file lines
    def ParseFuncBody(self, line, Insts):
        raise NoParsingEffort

    # Split basic block from the list instruction
    def SplitBlocks(self, Insts):
        raise NoParsingEffort

    # Retrieve instruction ID
    def GetInstNum(self, line):
        items = line.split('/*')
        return items[1]
    
    def ExtractPredicateReg(self, opcode):
        if opcode[0] == '@' and opcode[1] == 'P' and opcode[2].isdigit():
            return opcode[1:3]
        
        return None
    
    def ExtractPredicateRegNeg(self, opcode):
        if opcode[0] == '@' and opcode[1] == '!' and opcode[2] == 'P' and opcode[3].isdigit():
            return opcode[1:4]
            
        return None
    
    # Retrieve instruction's opcode
    def GetInstOpcode(self, line):
        items = line.split(';')
        line = (items[0].lstrip())
        items = line.split(' ')
        # Get opcode
        opcode = items[0]
        PFlag = None
        # Handle the condntion branch flags
        # if opcode == "@P0":
        #     PFlag = "P0"
        #     opcode = items[1]
        # elif opcode == "@!P0":
        #     PFlag = "!P0"
        #     opcode = items[1]
        pred_reg = self.ExtractPredicateReg(opcode)
        pred_reg_neg = self.ExtractPredicateRegNeg(opcode)
        rest_content = line.replace(items[0], "")
        if pred_reg:
            PFlag = pred_reg 
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")
        elif pred_reg_neg:
            PFlag = pred_reg_neg
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")
            
        return opcode, PFlag, rest_content

    # Retrieve instruction's operands
    def GetInstOperands(self, line):
        items = line.split(',')
        ops = []
        for item in items:
            operand = item.lstrip()
            if (operand != ''):
                ops.append(operand)

        return ops 

    # Parse instruction, includes operators and operands
    def ParseInstruction(self, InstID, Opcode_Content, PFlag, Operands_Content, Operands_Detail, CurrFunc):
        # Parse opcodes
        Opcodes = Opcode_Content.split('.')
        if PFlag != None:
            Opcodes.insert(0, PFlag)

        # Parse operands
        Operands = []
        for Operand_Content in Operands_Content:
            Operands.append(self.ParseOperand(Operand_Content, CurrFunc))

        # Create instruction
        return Instruction(InstID, Opcodes, Operands, Opcode_Content + " " + Operands_Detail)

    # Parse operand
    def ParseOperand(self, Operand_Content, CurrFunc):
        Operand_Content = Operand_Content.lstrip()
        IsReg = False
        Reg = None
        Name = None
        Suffix = None
        ArgOffset = -1
        IsArg = False
        IsDim = False
        IsThreadIdx = False

        # Check if it is a register for address pointer, e.g. [R0]
        if Operand_Content.find(PTRREG_PREFIX) == 0: # operand starts from '[R'
            # Fill out the ptr related charactors
            Operand_Content = Operand_Content.replace(PTR_PREFIX, "")
            Operand_Content = Operand_Content.replace(PTR_SUFFIX, "")
            
        # Check if it is a register
        if Operand_Content.find(REG_PREFIX) == 0: # operand starts from 'R' 
            IsReg = True
            Reg = Operand_Content
            Name = Operand_Content
            
            # Get the suffix of given operand
            items = Operand_Content.split('.')
            if len(items) > 1:
                Reg = items[0]
                Suffix = items[1]
                if len(items) > 2 and items[2] != "reuse": # reuse is a assembly hint for register reuse
                        raise InvalidOperandException
                
            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsDim, IsThreadIdx)

        # Check if it is a jump flag
        if Operand_Content.find(PATH_PREFIX) == 0: # operand starts from 'P'
            IsReg = True
            Reg = Operand_Content
            Name = Operand_Content

            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsDim, IsThreadIdx)
        
        # Check if it is a function argument or dimension related value
        if Operand_Content.find(ARG_PREFIX) == 0: 
            IsArg = True

            # Get the suffix of given operand
            items = Operand_Content.split('.')
            if len(items) > 1:
                Suffix = items[1]
                if len(items) > 2:
                    raise InvalidOperandException

                Operand_Content = items[0]
        
            ArgOffset = self.GetArgOffset(Operand_Content.replace(ARG_PREFIX, ""))
            if ArgOffset < ARG_OFFSET:
                IsArg = False
                IsDim = True

            # Create argument operaand
            Arg = Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsDim, IsThreadIdx)

            # Register argument to function
            CurrFunc.RegisterArg(ArgOffset, Arg)

            return Arg
        
        # Check if it is a special value
        items = Operand_Content.split('.')
        if len(items) == 2:
            if items[0].find(THREAD_IDX) == 0:                
                IsThreadIdx = True
                Suffix = items[1]

        return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsDim, IsThreadIdx)

    # Parse the argument offset
    def GetArgOffset(self, offset):
        offset = offset.replace('[', "")
        offset = offset.replace(']', "")
        return int(offset, base = 16)

    # Parse control code 
    def ParseControlCode(self, Content, ControlCodes):
       raise NoParsingEffort
   
    # Build the control-flow graph
    def BuildCFG(self, Blocks):
        # No need to process single basic block case
        if len(Blocks) == 1:
            return Blocks

        # Handle the multi-block case
        JumpTargets = {}
        for i in range(len(Blocks)):
            CurrBB = Blocks[i]
            # Handle branch target case: the branch instruciton locates in current basic block
            self.CheckAndAddTarget(CurrBB, CurrBB.GetBranchTarget(), JumpTargets)
 
            # Handle the direct target case: the next basic block contains exit instruction
            if i < len(Blocks) - 1:
                self.CheckAndAddTarget(CurrBB, CurrBB.GetDirectTarget(Blocks[i + 1]), JumpTargets)

        MergedTo = {}
        NewBlocks = []
        CurrBB = None
        for i in range(len(Blocks)):
            NextBB = Blocks[i]
            # Add CFG connection to its jump source
            if NextBB.addr in JumpTargets:
                for TargetBB in JumpTargets[NextBB.addr]:
                    if TargetBB in MergedTo:
                        TargetBB = MergedTo[TargetBB]
                    
                    TargetBB.AddSucc(NextBB)
                    NextBB.AddPred(TargetBB)
                # Reset current basic block, i.e. restart potential merging
                CurrBB = None
                
            # Handle the basic block merge case
            if CurrBB != None and NextBB.addr not in JumpTargets:
                # Merge two basic blocks
                CurrBB.Merge(NextBB)
                MergedTo[NextBB] = CurrBB

                continue
            
            if CurrBB == None:
                # Reset current basic block
                CurrBB = NextBB
                # Add current basic block to translated block list
                NewBlocks.append(CurrBB)

        for NewBB in NewBlocks:
            NewBB.EraseRedundency()
            # NewBB.dump()
                
        return NewBlocks

    #Create the control-flow graph
    def CreateCFG(self, Blocks):
        # No need to process single basic block case
        if len(Blocks) == 1:
            return Blocks

        for BB in Blocks:
            BB.EraseRedundency()
            # BB.dump()

        return Blocks
    
    # Check if the target address is legal, then add the target address associated with its jump source
    def CheckAndAddTarget(self, CurrBB, TargetAddr, JumpTargets):
        if TargetAddr > 0:
            if TargetAddr not in JumpTargets:
                JumpTargets[TargetAddr] = []
            JumpTargets[TargetAddr].append(CurrBB)

    # Create def-use chain
    def CreateDefUse(self, Insts):
        DU = DefUse()

        # The table of current definition
        CurrDefs = {}
        
        # Iterate through instructions
        #for Inst in Insts:
            
            # Check the use and connect the defs
            
            # Get def and put on current defs table

            #Inst.controlcode.dump()
