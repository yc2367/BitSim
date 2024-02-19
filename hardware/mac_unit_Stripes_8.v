`ifndef __mac_unit_Stripes_8_V__
`define __mac_unit_Stripes_8_V__

`include "max_comparator.v"

module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (sign) begin
			out = ~in + 1'b1;
		end else begin
			out = in;
		end
	end
endmodule


module value_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         en,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (en) begin
			out = in;
		end else begin
			out = 0;
		end
	end
endmodule


module mac_unit_Stripes_8
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = 2*DATA_WIDTH + 7,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             is_pooling,

	input  logic signed [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0], 
	input  logic                             w_bit    [VEC_LENGTH-1:0],
	input  logic                             is_msb,
	input  logic                             delayed_is_msb,
	input  logic signed [RESULT_WIDTH-1:0]   result_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_out  [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			value_select #(DATA_WIDTH) bs_mul (
				.in(act_in[j]), .en(w_bit[j]), .out(act_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+0:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+1:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_total;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = act_out[2*j] + act_out[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_total = psum_2[2*j] + psum_2[2*j+1];
		end
	endgenerate

	logic signed [DATA_WIDTH+2:0]  psum_total_true, psum_total_true_reg;
	pos_neg_select #(DATA_WIDTH+3) twos_complement (.in(psum_total), .sign(is_msb), .out(psum_total_true));

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	localparam PAD_WIDTH = ACC_WIDTH - RESULT_WIDTH;
	always_comb begin
		if (delayed_is_msb) begin
			accum_in = result_prev;
		end else begin
			accum_in = accum_out <<< 1;
		end
	end

	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
		end else if	(en) begin
			psum_total_true_reg <= psum_total_true;
			accum_out <= psum_total_true_reg + accum_in;
		end
	end

	logic signed [RESULT_WIDTH-1:0] comp_result;
	max_comparator #(RESULT_WIDTH) comp (
		.in_1(accum_out[ACC_WIDTH-1:ACC_WIDTH-16]), .in_2(result_prev), .out(comp_result)
	);

	always_comb begin
		if (is_pooling) begin
			result = comp_result;
		end else begin
			result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];
		end
	end
endmodule


module mac_unit_Stripes_8_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter ACC_WIDTH     = 2*DATA_WIDTH + 7,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             is_pooling,

	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0], 
	input  logic                             w_bit    [VEC_LENGTH-1:0],
	input  logic                             is_msb,
	input  logic                             delayed_is_msb,
	input  logic signed [RESULT_WIDTH-1:0]   result_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;
	
	logic signed [DATA_WIDTH-1:0]  act_in [VEC_LENGTH-1:0];
	generate
	for (j=0; j<VEC_LENGTH; j=j+1) begin
		always @(posedge clk) begin
			if (reset) begin
				act_in[j] <= 0;
			end else begin
				act_in[j] <= act[j];
			end
		end
	end
	endgenerate

	mac_unit_Stripes_8 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif