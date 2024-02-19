`ifndef __mac_unit_Pragmatic_16_V__
`define __mac_unit_Pragmatic_16_V__

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


module shifter #(
	parameter IN_WIDTH  = 11,
	parameter OUT_WIDTH = IN_WIDTH + 7
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [2:0]           shift_sel,
	input  logic                        en, 	
	output logic signed [OUT_WIDTH-1:0] out
);
	logic signed [OUT_WIDTH-1:0] out_tmp;
	always_comb begin 
		case (shift_sel)
			3'b000 : out_tmp = in;
			3'b001 : out_tmp = in <<< 1;
			3'b010 : out_tmp = in <<< 2;
			3'b011 : out_tmp = in <<< 3;
			3'b100 : out_tmp = in <<< 4;
			3'b101 : out_tmp = in <<< 5;
			3'b110 : out_tmp = in <<< 6;
			3'b111 : out_tmp = in <<< 7;
			default: out_tmp = {OUT_WIDTH{1'bx}};
		endcase
	end

	always_comb begin
		if (en) begin
			out = out_tmp;
		end else begin
			out = 0;
		end
	end
endmodule


module mac_unit_Pragmatic_16
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
	input  logic        [2:0]                w_idx    [VEC_LENGTH-1:0],
	input  logic                             w_en     [VEC_LENGTH-1:0],
	input  logic                             is_neg   [VEC_LENGTH-1:0],
	input  logic signed [RESULT_WIDTH-1:0]   result_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_tmp    [VEC_LENGTH-1:0] ;
	logic signed [DATA_WIDTH+6:0]  act_out    [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			pos_neg_select #(DATA_WIDTH) pos_neg_act (
				.in(act_in[j]), .out(act_tmp[j]), .sign(is_neg[j])
			);
			shifter #(DATA_WIDTH, DATA_WIDTH+7) shift_act (
				.in(act_tmp[j]), .shift_sel(w_idx[j]), .en(w_en[j]), .out(act_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+7:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+8:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+9:0]  psum_3 [VEC_LENGTH/8-1:0];
	logic signed [DATA_WIDTH+9:0]  psum_total;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = act_out[2*j] + act_out[2*j+1];
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
	localparam PAD_WIDTH = ACC_WIDTH - RESULT_WIDTH;
	always_comb begin
		if (load_accum) begin
			accum_in = {result_prev, {PAD_WIDTH{1'b0}}};
		end else begin
			accum_in = accum_out;
		end
	end

	logic signed [DATA_WIDTH+9:0]  psum_total_reg;
	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
		end else if	(en) begin
			psum_total_reg <= psum_total;
			accum_out <= psum_total_reg + accum_in;
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


module mac_unit_Pragmatic_16_clk
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
	input  logic        [2:0]                w_idx    [VEC_LENGTH-1:0],
	input  logic                             w_en     [VEC_LENGTH-1:0],
	input  logic                             is_neg   [VEC_LENGTH-1:0],
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

	mac_unit_Pragmatic_16 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif