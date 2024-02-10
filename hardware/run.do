vlog +acc  shifter.v shifter_tb.v 
vsim -voptargs=+acc work.shifter_tb
run -all 
quit -sim
rm transcript
exit