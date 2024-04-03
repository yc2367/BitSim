set tsmc28 "/opt/PDKs/TSMC/28nm/Std_Cell_Lib/tcbn28hpcplusbwp30p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp30p140_180a"
set workdir /home/yc2367/Research/BitSim/hardware
set_app_var target_library "$tsmc28/tcbn28hpcplusbwp30p140ssg0p9vm40c.db"
set_app_var link_library   "* $target_library"

set run_module 0 ;
set bit_func   0 ;
set group_size 16 ;

set add_load 1 ;

if {$run_module == 1} {
    analyze -format sverilog $workdir/scheduler_bitwave.v
    elaborate scheduler_bitwave
    if {$add_load == 1} {
        set_load 0.02 [all_outputs]
    }
} else {
    if {$bit_func == 0} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Vert_16.v
            elaborate mac_unit_Vert_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Vert_8.v
            elaborate mac_unit_Vert_8_clk
        }
    } elseif {$bit_func == 1} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Wave_16.v
            elaborate mac_unit_Wave_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Wave_8.v
            elaborate mac_unit_Wave_8_clk
        }
    } elseif {$bit_func == 2} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Bitlet_16.v
            elaborate mac_unit_Bitlet_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Bitlet_32.v
            elaborate mac_unit_Bitlet_32_clk
        }
    } elseif {$bit_func == 3} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Pragmatic_16.v
            elaborate mac_unit_Pragmatic_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Pragmatic_8.v
            elaborate mac_unit_Pragmatic_8_clk
        }
    } elseif {$bit_func == 4} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Stripes_16.v
            elaborate mac_unit_Stripes_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Stripes_8.v
            elaborate mac_unit_Stripes_8_clk
        }
    } elseif {$bit_func == 5} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Parallel.v
            elaborate mac_unit_Parallel_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Parallel_64.v
            elaborate mac_unit_Parallel_64_clk
        }
    } elseif {$bit_func == 6} {
        if {$group_size == 16} {
            analyze -format sverilog $workdir/mac_unit_Precompute_16.v
            elaborate mac_unit_Precompute_16_clk
        } else {
            analyze -format sverilog $workdir/mac_unit_Precompute_8.v
            elaborate mac_unit_Precompute_8_clk
        }
    }
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