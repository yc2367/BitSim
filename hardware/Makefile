
# modify the following for your design
TOP_DESIGN		= post.28nm.syn
CLOCK_PERIOD	= 1.25	# in the unit of ns
CLOCK_PORT		= clk
RESET_PORT		= reset
SIM_TIMESCALE	= 1ns/1ps
HEADERS			= $(wildcard rtl/*.svh) # It's recommended to  put all the macro/interface/parameter definition in header file
# TESTBENCH		= $(wildcard tb/verilog/tb_osp.sv tb/verilog/cell_model.sv)
TESTBENCH		= $(wildcard ./tb_bitvert/vert_32_tb.v)
# SYN_FILE		= $(wildcard rtl/os_ws_comp.sv)
SYN_FILE		= $(wildcard rtl_tapeout/*.sv)
CELL_MODEL		= $(wildcard tb/verilog/SP018*.v) $(wildcard /home/hxf/PDK/SMIC/SCC018V3EBCD_UHD_RVT_V0p1c/verilog/scc018v3ebcd_uhd_rvt.v)
IO_MODEL		= $(wildcard tb/verilog/SP018V3EBCD*.v)
NETLIST			= ./work/$(TOP_DESIGN).v
SDC_FILE		= ./work/$(TOP_DESIGN).sdc
# SIM_FILE		= $(wildcard rtl_tapeout/*.sv) $(TESTBENCH)
SIM_FILE		= $(wildcard ./tb_bitvert/*.v) 
SYNTH_DIR 		= ./syn

# export file
export HEADERS
export SYN_FILE
export TOP_DESIGN
export CLOCK_PERIOD
export CLOCK_PORT
export RESET_PORT
export NETLIST
export SDC_FILE

VCS = vcs -V -sverilog +v2k -full64 -debug_acc+all +vcs+dumpvars+test.vcd +lint=TFIPC-L -debug_region+cell+encrypt -l com.log 
# -P ${NOVAS_HOME}/share/PLI/VCS/LINUX64/novas.tab ${NOVAS_HOME}/share/PLI/VCS/LINUX64/pli.a
LIB = /opt/PDKs/TSMC/28nm/Std_Cell_Lib/tcbn28hpcplusbwp30p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp30p140_110a/tcbn28hpcplusbwp30p140.v

# Simulation
all:	simv
	./simv | tee program.out
	exit

simv: $(TESTBENCH) #$(CELL_MODEL)
	$(VCS) -override_timescale=$(SIM_TIMESCALE) $^ -o simv

verdi:	simv
	verdi -full64 -ssf test.fsdb -F ./tb/verdifile.f

.PHONY:	all simv

# Lint
lint:	$(HEADERS) $(SYN_FILE)
	cd tb ; spyglass -shell -project lint.prj


# Synthesis 

link:	$(HEADERS) $(SYN_FILE) 
	cd $(SYNTH_DIR) && dc_shell-t -f ./link.tcl | tee syn.out


syn:	$(HEADERS) $(SYN_FILE) 
	cd $(SYNTH_DIR) && dc_shell-t -f ./syn.tcl | tee syn.out

syn_simv:	$(CELL_MODEL) $(NETLIST) $(TESTBENCH)
	$(VCS) $^ $(LIB) +define+SYNTH_TEST -diag timescale -override_timescale=$(SIM_TIMESCALE) -o syn_simv
	

verdi_syn:	syn_simv
	./syn_simv | tee syn_simv.out
	verdi -full64 -ssf test.fsdb -F ./syn/verdifile.f

.PHONY:	syn syn_simv verdi_syn 

# STA/PT
pt:		$(NETLIST)	$(SDC_FILE)
	cd $(SYNTH_DIR) && pt_shell -f ./pt.tcl | tee pt.out

clean:
	rm -rf *.log csrc simv* *.key *.vpd DVEfile *~ *.fsdb* novas* *Log 
	rm -rf *simv program.out csrc *simv.daidir syn*.out
	rm -rf syn/*.rpt syn/alib* syn/*.svf syn/*.log syn/*.mr syn/*.out
	
nuke:	clean
	rm -rf syn/data/* syn/reports/* syn/work/* 
	rm ucli.key simv program.out
