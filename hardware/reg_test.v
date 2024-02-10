`ifndef __reg_test_V__
`define __reg_test_V__

module reg_test
#(
    parameter DATA_WIDTH = 8
) (
	input  logic                          clk,
	input  logic                          reset,
	input  logic signed [DATA_WIDTH-1:0]  in  [7:0],
	output logic signed [DATA_WIDTH-1:0]  out [7:0]
);
	localparam VEC_LENGTH = 8;
	genvar j;

	generate
	for (j=0; j<VEC_LENGTH; j=j+1) begin
		always @(posedge clk) begin
			if (reset) begin
				out[j] <= 0;
			end else begin
				out[j] <= in[j];
			end
		end
	end
	endgenerate

endmodule

`endif