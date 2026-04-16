from parse.parser_sm52 import SaSSParser_SM52
from parse.parser_sm35 import SaSSParser_SM35
from parse.parser_sm75 import SaSSParser_SM75

class InvalidISAException(Exception):
    pass

class SaSSParser:
    def __init__(self, file):
        # Retrieve ISA
        isa = self.get_isa(file)
        # print("parse ", isa)

        self.file = file
        if isa == "sm_52":
            self.parser = SaSSParser_SM52(isa, file)
        elif isa == "sm_35":
            self.parser = SaSSParser_SM35(isa, file)
        elif isa == "sm_75":
            self.parser = SaSSParser_SM75(isa, file)
        else:
            raise InvalidISAException
            
    # Retrieve ISA definition
    def get_isa(self, file):
        for line_num, line in enumerate(file.split('\n')):
            if "arch = sm_" in line :
                items = line.split(" = ")
                return items[1]

        raise InvalidISAException
        
    # Parse the SaSS text file
    def apply(self):
        return self.parser.apply()
    
