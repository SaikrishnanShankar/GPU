#h# ***** Makefile to build Macsim and run benchmarks *****

# .SILENT:

# Environment check
ifeq (${MACSIM_DIR},)
    $(error "'MACSIM_DIR' environment variable not set; did you forget to source the sourceme script?")
endif

CC        := g++
LD        := g++
CXXFLAGS  := -Wall -Wno-unused-variable -Wno-unused-but-set-variable -std=c++11 -I src/macsim
LDFLAGS   := -lz

ifeq ($(DEBUG),1)
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O3
endif

ifeq ($(COV),1)
    CXXFLAGS+= -fPIC -fprofile-arcs -ftest-coverage
endif

MODULES   := macsim exec utils
SRC_DIR   := $(addprefix src/,$(MODULES)) src
BUILD_DIR := $(addprefix build/,$(MODULES)) build

SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC       := src/main.cpp $(SRC)
OBJ       := $(patsubst src/%.cpp,build/%.o,$(SRC))
DEP       := $(OBJ:.o=.d)
INCLUDES  := $(addprefix -I,$(SRC_DIR))

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	@printf "> Compiling	$$(@F)\n"
	$(CC) $(CXXFLAGS) $(INCLUDES) -MMD -c $$< -o $$@
endef

OUTPUT_TAR	:= 	CS8803proj3_${USER}.tar
LOG_DIR		:=	${MACSIM_DIR}/log
TRACE_DIR	:=	${MACSIM_TRACE_DIR}

$(shell mkdir -p $(LOG_DIR))

.PHONY: all checkdirs clean

all: checkdirs macsim						#t# Build macsim

macsim: $(OBJ)
	@printf "> Linking 	$(@F)\n"
	$(LD) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

.PHONY: cov
cov: $(SRC)									#t# generate coverage metadata
	gcov $^ -o build

.PHONY: traces
traces:										#t# Download traces (if not present)
	bash ./scripts/get_traces.sh $(TRACE_DIR)

.PHONY: run_task0
run_task0: traces							#t# Run simulator with the default (RR) warp scheduling policy
	python3 scripts/runner.py -c RR -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: run_task1
run_task1: traces							#t# Run simulator for task-1 (GTO)
	python3 scripts/runner.py -c RR:GTO -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: run_task2
run_task2: traces							#t# Run simulator for task-2 (GTO and CCWS)
	python3 scripts/runner.py -c RR:GTO:CCWS -t $(TRACE_DIR) -l $(LOG_DIR) -r -d $(LOG_DIR)/results.txt -j $(LOG_DIR)/results.json

.PHONY: plot
plot:										#t# Plot stats
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json NUM_STALL_CYCLES
	python3 scripts/plot.py -t $(TRACE_DIR) -l $(LOG_DIR) $(LOG_DIR)/results.json MISSES_PER_1000_INSTR


.PHONY: submit
submit:	clean								#t# Generate submission tarball
	@echo "Generating tar..."
	tar --exclude=.python3 \
	--exclude=.vscode \
	--exclude=build \
	--exclude=log \
	--exclude=*.zip \
	--exclude=*.tar* \
	--exclude=*.png \
	--exclude=*__pycache__* \
	--exclude=macsim_traces \
	-czhvf $(OUTPUT_TAR) *
	


.PHONY: clean
clean:										#t# Clean build files
	rm -rf $(BUILD_DIR)
	rm -f macsim
	rm -f *.log
	rm -f $(OUTPUT_TAR)
	rm -f *.gcov


.PHONY: clean-logs
clean-logs:									#t# Clean logs (log dir)
	rm -rf $(LOG_DIR)/*


.PHONY: clean-all
clean-all: clean clean-logs					#t# Clean build files and logs

.PHONY: help
help: Makefile								#t# Print help
# Print Header comments '#h#'
	@sed -n 's/^#h# //p' $<

# Print Target descriptions '#t#'
	@printf "\nTARGETs:\n"
	@grep -E -h '\s#t#\s' $< | awk 'BEGIN {FS = ":.*?#t# "}; {printf "\t%-20s %s\n", $$1, $$2}'

# Print footer '#f#'
	@sed -n 's/^#f# //p' $<


-include $(DEP)
$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
