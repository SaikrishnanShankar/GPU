from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.operand import Operand
from sir.operand import InvalidOperandException

from parse.parser_base import SaSSParserBase
from parse.parser_base import UnmatchedControlCode

SM52_CTLCODE_LEN = 3

class SaSSParser_SM52(SaSSParserBase):
    def __init__(self, isa, file):
        self.file = file
        self.CtlCodes = []

    # Parse the SaSS text file
    def applyx(self):
        # List of functions
        Funcs = []
        # List of basic blocks in current function
        Blocks = []
    
        # If the current loop iteration is parsing function
        IsParsingFunc = False
        # If the current loop iteration is parsing basic block
        IsParsingBB = False
        
        # Current function
        CurrFunc = None
        # Current basic block
        CurrBB = None
        # Current control code associated with basic block
        CurrCtrCode = None
        
        # Main loop that parses the SaSS text file
        for line_num, line in enumerate(self.file.split('\n')):
            # Process lines in SaSS text file

            # Process function title and misc
            if (not ("/*" in line and "*/" in line)):
                # Check function start
                if ("Function : " in line):
                    # Wrap up previous function
                    if IsParsingFunc and CurrFunc != None:
                        CurrFunc.blocks = self.CreateCFG(Blocks)
                        # Reset current function, list of basic blocks and instructions
                        Func = None
                        Blocks = []
                
                    items = line.split(' : ')

                    # Create new function
                    CurrFunc = Function(items[1])
                    Funcs.append(CurrFunc)
                    
                    # Setup the flags that for parsing function
                    IsParsingFunc = True
                continue

            # Process function body 
            items = line.split('*/')
            if len(items) == 2:
                # This is the interval between code sections, and represent control branch
                IsParsingBB = False

                # Parse the control code
                CurrCtrCodes = self.ParseControlCode(items[0], self.CtlCodes)
                
                continue
            elif len(items) == 3 and not IsParsingBB:
                # Set the flag to start a new basic block 
                IsParsingBB = True

                # Create a new basic block
                CurrBB = BasicBlock(self.GetInstNum(items[0]), CurrCtrCodes)
                CurrBB.ControlCodes = CurrCtrCodes
                Blocks.append(CurrBB)
                
            # Retrieve instruction ID
            inst_id = self.GetInstNum(items[0])
            # Retrieve instruction opcode
            inst_opcode, rest_content = self.GetInstOpcode(items[1])
            rest_content = rest_content.replace(" ", "")
            if (rest_content == "EXIT"):
                # Special case for exit instruction
                inst_ops = inst_opcode
                inst_opcode = rest_content
            else:
                # Retrieve instruction operands
                inst_ops = self.GetInstOperands(rest_content)
            
            # Create instruction
            inst = self.ParseInstruction(inst_id, inst_opcode, inst_ops, rest_content, CurrFunc)

            # Add instruction into list
            CurrBB.AppendInst(inst)

        # Wrap up previous function
        if IsParsingFunc and CurrFunc != None:
            CurrFunc.blocks = self.CreateCFG(Blocks)

        CurrFunc.DumpCFG()
        
        return Funcs

    # Parse the function body from file lines
    def ParseFuncBody(self, line, Insts, CurrFunc):
        # Process function body 
        items = line.split('*/')
        if len(items) == 2:
            # Parse the control code
            self.ParseControlCode(items[0], self.CtlCodes)
        elif len(items) == 3:    
            # Retrieve instruction ID
            inst_id = self.GetInstNum(items[0])
            # Retrieve instruction opcode
            inst_opcode, pflag, rest_content = self.GetInstOpcode(items[1])
            rest_content = rest_content.replace(" ", "")

            # Retrieve instruction operands
            inst_ops = self.GetInstOperands(rest_content)
            
            # Create instruction
            inst = self.ParseInstruction(inst_id, inst_opcode, pflag, inst_ops, rest_content, CurrFunc)

            # Add control code
            if len(self.CtlCodes) > 0:
                CtlCode = self.CtlCodes[0]
                inst.SetCtlCode(CtlCode)
                # Remove the control code from temprory storage
                self.CtlCodes.remove(CtlCode)
            else:
                raise UnmatchedControlCode
            
            # Add instruction into list
            Insts.append(inst)
            
    
    # Parse control code 
    def ParseControlCode(self, Content, ControlCodes):
        Content = Content.replace("/*", "")
        Content = Content.replace(" ", "")

        # This parsing process is correspoinding to Maxwell architecture, which has
        # 3 17-bits sections for control and 3 4-bits sections for reuse

        # Extract control sections
        Sec1 = Content[2 : 8]
        Sec2 = Content[8 : 13]
        Sec3 = Content[13 : ]
        Sec1Num = int("0x" + Sec1, 16) & int("0x7fffc", 16)
        Sec2Num = int("0x" + Sec2, 16) & int("0x3fffe", 16)
        Sec3Num = int("0x" + Sec3, 16) & int("0x1ffff", 16)

        Stall = (Sec3Num & 15) >> 0
        Yield = (Sec3Num & 16) >> 4
        WrtB  = (Sec3Num & 224) >> 5
        ReadB = (Sec3Num & 1792) >> 8
        WaitB = (Sec3Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))
        
        Stall = (Sec2Num & 15) >> 0
        Yield = (Sec2Num & 16) >> 4
        WrtB  = (Sec2Num & 224) >> 5
        ReadB = (Sec2Num & 1792) >> 8
        WaitB = (Sec2Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))
        
        Stall = (Sec1Num & 15) >> 0
        Yield = (Sec1Num & 16) >> 4
        WrtB  = (Sec1Num & 224) >> 5
        ReadB = (Sec1Num & 1792) >> 8
        WaitB = (Sec1Num & 129024) >> 11
        ControlCodes.append(ControlCode(Content, WaitB, ReadB, WrtB, Yield, Stall))

        return ControlCodes
 
    # Split basic block from the list instruction
    def SplitBlocks(self, Insts):
        BBs = []
        BranchBBs = []
        
        # Initialize current basic block
        CurrBB = None
        PredBB = None
        BranchBB = None

        for Inst in Insts:
            if Inst.InCondPath():
                if not BranchBB.IsInitialized():
                    # Create the new branch basic block
                    BranchBB.Init(Inst.id, Inst.pflag)

                # Add instruction into branch basic block
                BranchBB.AppendInst(Inst)
                
            else:
                if CurrBB == None:
                    # Create the new basic block
                    CurrBB = BasicBlock(Inst.id, None)
                    BBs.append(CurrBB)

                    # Add branch connection
                    if PredBB != None:
                        # Setup cpnnection
                        PredBB.AddSucc(CurrBB)
                        CurrBB.AddPred(PredBB)

                # Add instruction into main basic block
                CurrBB.AppendInst(Inst)

                if Inst.IsBranch():
                    # Set predecessor
                    PredBB = CurrBB

                    # Cleanup current basic block and branch basic block
                    CurrBB = None

                    if BranchBB != None and BranchBB.IsEmpty():
                        # The previous branch instruction does not make corresponding branch BB yet, so it will follow with current branch instruction
                        # Setup the predecessor and successor for branch basic block
                        PredBB.AddSucc(BranchBB)
                        BranchBB.AddPred(PredBB)
                    else:
                        # Create an empty basic block
                        BranchBB = BasicBlock(None, None)
                        BranchBBs.append(BranchBB)

                        # Setup the predecessor and successor for branch basic block
                        PredBB.AddSucc(BranchBB)
                        BranchBB.AddPred(PredBB)

        for BranchBB in BranchBBs:
            BBs.append(BranchBB)
            
        return BBs
