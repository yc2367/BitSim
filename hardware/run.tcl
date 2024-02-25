set tsmc28 /pdks/tsmc/tsmc-28nm-cln28hpc-nda/stdview
set workdir /home/yc2367/Desktop/Research/BitSim/hardware
set_app_var target_library "$tsmc28/stdcells.db"
set_app_var link_library   "* $target_library"

set func 0 ;
if {$func == 0} {
    analyze -format sverilog $workdir/mac_unit_Vert_16.v
    elaborate mac_unit_Vert_16_clk
} elseif {$func == 1} {
    analyze -format sverilog $workdir/mac_unit_Wave_8.v
    elaborate mac_unit_Wave_8_clk
} elseif {$func == 2} {
    analyze -format sverilog $workdir/mac_unit_Pragmatic_16.v
    elaborate mac_unit_Pragmatic_16_clk
} elseif {$func == 3} {
    analyze -format sverilog $workdir/mac_unit_Stripes_16.v
    elaborate mac_unit_Stripes_16_clk
} elseif {$func == 4} {
    analyze -format sverilog $workdir/mac_accumulator_config_clk.v
    elaborate mac_accumulator_config_clk
}

check_design
create_clock clk -name ideal_clock1 -period 1.25
compile

write -format verilog -hierarchy -output post-synth.v
write -format ddc     -hierarchy -output post-synth.ddc
report_resources -nosplit -hierarchy
report_timing -nosplit -transition_time -nets -attributes
report_area -nosplit -hierarchy
report_power -nosplit -hierarchy

exit