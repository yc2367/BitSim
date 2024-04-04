vlog +acc  mac_unit_Vert_32_no_mul.v vert_32_tb.v 
vsim -voptargs=+acc work.vert_32_tb
run -all 
quit -sim
rm transcript
exit