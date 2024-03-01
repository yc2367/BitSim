set tsmc28 /pdks/tsmc/tsmc-28nm-cln28hpc-nda/stdview
set workdir /home/yc2367/Desktop/Research/BitSim/hardware
set_app_var target_library "$tsmc28/stdcells.db"
set_app_var link_library   "* $target_library"

set func 0 ;
set group_size 16 ;

if {$func == 0} {
    if {$group_size == 16} {
        analyze -format sverilog $workdir/mac_unit_Vert_16.v
        elaborate mac_unit_Vert_16_clk
    } else {
        analyze -format sverilog $workdir/mac_unit_Vert_8.v
        elaborate mac_unit_Vert_8_clk
    }
} elseif {$func == 1} {
    if {$group_size == 16} {
        analyze -format sverilog $workdir/mac_unit_Wave_16.v
        elaborate mac_unit_Wave_16_clk
    } else {
        analyze -format sverilog $workdir/mac_unit_Wave_8.v
        elaborate mac_unit_Wave_8_clk
    }
} elseif {$func == 2} {
    if {$group_size == 16} {
        analyze -format sverilog $workdir/mac_unit_Pragmatic_16.v
        elaborate mac_unit_Pragmatic_16_clk
    } else {
        analyze -format sverilog $workdir/mac_unit_Pragmatic_8.v
        elaborate mac_unit_Pragmatic_8_clk
    }
} elseif {$func == 3} {
    if {$group_size == 16} {
        analyze -format sverilog $workdir/mac_unit_Stripes_16.v
        elaborate mac_unit_Stripes_16_clk
    } else {
        analyze -format sverilog $workdir/mac_unit_Stripes_8.v
        elaborate mac_unit_Stripes_8_clk
    }
} elseif {$func == 4} {
    if {$group_size == 16} {
        analyze -format sverilog $workdir/mac_unit_Parallel_16.v
        elaborate mac_unit_Parallel_16_clk
    } else {
        analyze -format sverilog $workdir/mac_unit_Parallel_128.v
        elaborate mac_unit_Parallel_128_clk
    }
}

check_design
create_clock clk -name ideal_clock1 -period 1.2
compile

write -format verilog -hierarchy -output post-synth.v
write -format ddc     -hierarchy -output post-synth.ddc
report_resources -nosplit -hierarchy
report_timing -nosplit -transition_time -nets -attributes
report_area -nosplit -hierarchy
report_power -nosplit -hierarchy

exit