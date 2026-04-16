#ifndef GPU_PARAMETER_SET_H
#define GPU_PARAMETER_SET_H

#include "Parameter_Set_Base.h"
#include "macsim/macsim.h"

enum class Block_Scheduling_Policy_Types;
enum class Warp_Scheduling_Policy_Types;

class GPU_Parameter_Set : public Parameter_Set_Base
{
public:
	static int Cycle_Per_Period;
	static int Num_Of_Cores;
	static int Max_Block_Per_Core;
	static Block_Scheduling_Policy_Types Block_Scheduling_Policy;
	static Warp_Scheduling_Policy_Types Warp_Scheduling_Policy;
	static std::string GPU_Trace_Path;
	static int N_Repeat;
	static bool Enable_GPU_Cache;
	static bool GPU_Cache_Log;
	static int L1Cache_Size;
	static int L1Cache_Assoc;
	static int L1Cache_Line_Size;
	static int L1Cache_Banks;
	static int L2Cache_Size;
	static int L2Cache_Assoc;
	static int L2Cache_Line_Size;
	static int L2Cache_Banks;

	void XML_serialize(Utils::XmlWriter& xmlwriter);
	void XML_deserialize(rapidxml::xml_node<> *node);
};

#endif // !GPU_PARAMETER_SET_H
