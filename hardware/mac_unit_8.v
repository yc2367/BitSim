`ifndef __mac_unit_8_V__
`define __mac_unit_8_V__


module mac_unit_8
#(
    parameter DATA_WIDTH = 8
) (
	input  logic                          clk,
	input  logic                          reset,
	input  logic signed [DATA_WIDTH-1:0]  vec [7:0], 
	output logic signed [DATA_WIDTH+2:0]  result
);
	localparam VEC_LENGTH = 8;

	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_in [VEC_LENGTH-1:0] ;
	generate
	for (j=0; j<VEC_LENGTH; j=j+1) begin
		always @(posedge clk) begin
			if (reset) begin
				act_in[j] <= 0;
			end else begin
				act_in[j] <= vec[j];
			end
		end
	end
	endgenerate

	logic signed [DATA_WIDTH:0]    psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+1:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_3;
	generate
	for (j=0; j<VEC_LENGTH/2; j=j+1) begin
		assign psum_1[j] = act_in[2*j] + act_in[2*j+1];
	end

	for (j=0; j<VEC_LENGTH/4; j=j+1) begin
		assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
	end

	for (j=0; j<VEC_LENGTH/8; j=j+1) begin
		assign psum_3 = psum_2[2*j] + psum_2[2*j+1];
	end
	endgenerate

	always @(posedge clk) begin
		if (reset) begin
			result <= 0;
		end else begin
			result <= psum_3;
		end
	end

endmodule

`endif