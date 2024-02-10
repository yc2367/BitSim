`ifndef __shifter_V__
`define __shifter_V__

module shifter #(
	parameter IN_WIDTH  = 11,
	parameter OUT_WIDTH = 18
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [2:0]           shift_sel,	
	output logic signed [OUT_WIDTH-1:0] out
);
	assign out = in <<< shift_sel;
	
endmodule

`endif