`ifndef __mac_unit_Parallel_128_V__
`define __mac_unit_Parallel_128_V__


module mac_unit_Parallel_128
#(
    parameter DATA_WIDTH    = 8,
    parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             mul_en,
	input  logic                             acc_en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in  [VEC_LENGTH-1:0], 
	input  logic signed [DATA_WIDTH-1:0]     w_in    [VEC_LENGTH-1:0], 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [RESULT_WIDTH-1:0]  mul_out [VEC_LENGTH-1:0];
	logic signed [RESULT_WIDTH+0:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [RESULT_WIDTH+1:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [RESULT_WIDTH+2:0]  psum_3 [VEC_LENGTH/8-1:0];
	logic signed [RESULT_WIDTH+3:0]  psum_total, psum_total_reg;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			assign mul_out[j] = act_in[j] * w_in[j];
		end

		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			always @(posedge clk) begin
				if (reset) begin
					psum_1[j] <= 0;
				end else if	(mul_en) begin
			    	psum_1[j] <= mul_out[2*j] + mul_out[2*j+1];
				end
			end
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_3[j] = psum_2[2*j] + psum_2[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/16; j=j+1) begin
			assign psum_total = psum_3[2*j] + psum_3[2*j+1];
		end
	endgenerate

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	always_comb begin
		if (load_accum) begin
			accum_in = accum_prev;
		end else begin
			accum_in = accum_out;
		end
	end

	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
			psum_total_reg <= 0;
		end else if	(acc_en) begin
			psum_total_reg <= psum_total;
			accum_out <= psum_total_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Parallel_128_clk
#(
    parameter DATA_WIDTH    = 8,
    parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             mul_en,
	input  logic                             acc_en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act  [VEC_LENGTH-1:0], 
	input  logic signed [DATA_WIDTH-1:0]     w_in [VEC_LENGTH-1:0], 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_in  [VEC_LENGTH-1:0];
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			always @(posedge clk) begin
				if (reset) begin
					act_in[j] <= 0;
				end else if (mul_en) begin
					act_in[j] <= act[j];
				end
			end
		end
	endgenerate

	mac_unit_Parallel_128 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif