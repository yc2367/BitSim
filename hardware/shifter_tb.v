`ifndef __shifter_tb_V__
`define __shifter_tb_V__

`include "shifter.v"

module shifter_tb;

  parameter IN_WIDTH  = 12;
	parameter OUT_WIDTH = 19;

  logic signed [IN_WIDTH-1:0]  in;
  logic        [2:0]           shift_sel;
  logic signed [OUT_WIDTH-1:0] out;

  shifter #(IN_WIDTH, OUT_WIDTH) s (.*);

  initial
    $monitor ("in=%b, shift_sel=%b, out=%b", in, shift_sel, out);

  initial begin
    #0 in = 12'b111011110011;  shift_sel = 0;
    #3 shift_sel = 1;
    #3 shift_sel = 2;
    #3 shift_sel = 3;
    #3 shift_sel = 4;
    #3 shift_sel = 5;
    #3 shift_sel = 6;
    #3 shift_sel = 7;
    
    #15 $stop;
  end

endmodule

`endif