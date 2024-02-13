`ifndef __max_comparator_V__
`define __max_comparator_V__


module max_comparator #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0]  in_1,
	input  logic signed [DATA_WIDTH-1:0]  in_2,
	output logic signed [DATA_WIDTH-1:0]  out
); 
	always_comb begin
		if ( in_1 > in_2 ) begin
			out = in_1;
		end else begin
			out = in_2;
		end
	end
endmodule

`endif