`ifndef __mac_unit_16_Wave_V__
`define __mac_unit_16_Wave_V__

`include "pos_neg_select.v"
`include "max_comparator.v"

module value_select #(
	parameter DATA_WIDTH = 9
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         w_bit,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (w_bit) begin
			out = in;
		end else begin
			out = 0;
		end
	end
endmodule


module shifter #(
	parameter IN_WIDTH  = 12,
	parameter OUT_WIDTH = 19
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [2:0]           shift_sel,	
	output logic signed [OUT_WIDTH-1:0] out
);
	always_comb begin 
		case (shift_sel)
			3'b000 : out = in;
			3'b001 : out = in <<< 1;
			3'b010 : out = in <<< 2;
			3'b011 : out = in <<< 3;
			3'b100 : out = in <<< 4;
			3'b101 : out = in <<< 5;
			3'b110 : out = in <<< 6;
			3'b111 : out = in <<< 7;
			default: out = {OUT_WIDTH{1'bx}};
		endcase
	end
	
endmodule


module mac_unit_16_Wave
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,
	input  logic                             is_pooling,

	input  logic signed [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0], 
	input  logic                             sign     [VEC_LENGTH-1:0],
	input  logic                             w_bit    [VEC_LENGTH-1:0],
	input  logic        [2:0]                column_idx,
	input  logic signed [RESULT_WIDTH-1:0]   result_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH:0]  act_out    [VEC_LENGTH-1:0] ;
	logic signed [DATA_WIDTH:0]  bs_mul_out [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			pos_neg_select #(DATA_WIDTH) select_1 (
				.in(act_in[j]), .sign(sign[j]), .out(act_out[j])
			);
			value_select #(DATA_WIDTH+1) select_2 (
				.in(act_out[j]), .w_bit(w_bit[j]), .out(bs_mul_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+1:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+2:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+3:0]  psum_3 [VEC_LENGTH/8-1:0];
	logic signed [DATA_WIDTH+3:0]  psum_total ;

	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = bs_mul_out[2*j] + bs_mul_out[2*j+1];
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

	logic signed [DATA_WIDTH+10:0]  shifted_psum, shifted_psum_reg;
	shifter #(.IN_WIDTH(DATA_WIDTH+4), .OUT_WIDTH(DATA_WIDTH+11)) shift (
		.in(psum_total), .shift_sel(column_idx), .out(shifted_psum)
	);

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	localparam PAD_WIDTH = ACC_WIDTH - RESULT_WIDTH;
	always_comb begin
		if (load_accum) begin
			accum_in = {result_prev, {PAD_WIDTH{1'b0}}};
		end else begin
			accum_in = accum_out;
		end
	end

	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
		end else if	(en) begin
			shifted_psum_reg <= shifted_psum;
			accum_out <= shifted_psum_reg + accum_in;
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


module mac_unit_16_Wave_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,
	input  logic                             is_pooling,

	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0], 
	input  logic                             sign     [VEC_LENGTH-1:0],
	input  logic                             w_bit    [VEC_LENGTH-1:0],
	input  logic        [2:0]                column_idx,
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

	mac_unit_16_Wave #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif