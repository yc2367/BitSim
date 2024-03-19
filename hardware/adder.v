`ifndef __adder_V__
`define __adder_V__

module adder #(
	parameter IN_WIDTH  = 8
) (
	input  logic signed [IN_WIDTH-1:0]  in_1,
	input  logic signed [IN_WIDTH-1:0]  in_2,
	output logic signed [IN_WIDTH:0]    out
); 
	assign out = in_1 + in_2;
endmodule

`endif