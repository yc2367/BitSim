`ifndef __pos_neg_select_V__
`define __pos_neg_select_V__


module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH:0]   out
); 
	logic signed [DATA_WIDTH:0] in_extended;
	assign in_extended = {in[DATA_WIDTH-1], in};

	always_comb begin
		if (sign) begin
			out = ~in_extended + 1'b1;
		end else begin
			out = in_extended;
		end
	end
endmodule

`endif