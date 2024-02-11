`ifndef __mac_unit_16_Vert_2_module_V__
`define __mac_unit_16_Vert_2_module_V__

`include "mux_9to1.v"
`include "mux_17to1.v"
`include "mac_unit_adder_tree_8_Vert.v"

module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0]  in,
	input  logic                          sign,	
	output logic signed [DATA_WIDTH-1:0]  out
); 
	always_comb begin
		if (sign) begin
			out = ~in + 1'b1;
		end else begin
			out = in;
		end
	end
endmodule


module shifter_3bit #(
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


module shifter_constant #( // can only shift 3-bit or no shift
	parameter IN_WIDTH  = 12,
	parameter OUT_WIDTH = 15
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic                        is_shift,	
	output logic signed [OUT_WIDTH-1:0] out
);
	always_comb begin 
		if (is_shift) begin
			out = in <<< 3;
		end else begin 
			out = in;
		end
	end
endmodule


module mac_unit_16_Vert_2_module
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH) + 1,
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,

	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic        [MUX_SEL_WIDTH-2:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal

	// signal to select an activation that can be calculated wuth hamming distance 
	input  logic        [MUX_SEL_WIDTH-1:0]  hamming_sel,  
	input  logic                             hamming_sign,  

	input  logic signed [SUM_ACT_WIDTH-2:0]  sum_act [1:0], // sum of a group of activations (signed)
	input  logic        [2:0]                column_idx,    // current column index for shifting 
	input  logic        [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic                             is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                             is_msb,        // specify if the current column is MSB
	input  logic                             is_skip_zero [1:0],  // specify if skip bit 0
	
	output logic signed [DATA_WIDTH+16:0]    result
);
	genvar i, j;

	logic signed [DATA_WIDTH-1:0]     actin     [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	logic signed [DATA_WIDTH-1:0]     adder_in  [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	generate
		for (i=0; i<VEC_LENGTH/8; i=i+1) begin
			for (j=0; j<VEC_LENGTH/4; j=j+1) begin
				mux_9to1 #(DATA_WIDTH) mux_act (
					.vec(act[8*i+7:8*i]), .sel(act_sel[4*i+j]), .out(actin[4*i+j])
				);
			end
		end

		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			always @(posedge clk) begin
				if (reset) begin
					adder_in[j] <= 0;
				end else if (en) begin
					adder_in[j] <= actin[j];
				end
			end
		end
	endgenerate

	logic signed [SUM_ACT_WIDTH-2:0]    psum_mux_out  [1:0];
	mac_unit_adder_tree_8_Vert #(DATA_WIDTH, VEC_LENGTH/4, SUM_ACT_WIDTH-1) adder_tree_1 (
		.adder_in(adder_in[3:0]), .sum_act(sum_act[0]), .is_msb(is_msb), 
		.is_skip_zero(is_skip_zero[0]), .result(psum_mux_out[0])
	);
	mac_unit_adder_tree_8_Vert #(DATA_WIDTH, VEC_LENGTH/4, SUM_ACT_WIDTH-1) adder_tree_2 (
		.adder_in(adder_in[7:4]), .sum_act(sum_act[1]), .is_msb(is_msb), 
		.is_skip_zero(is_skip_zero[1]), .result(psum_mux_out[1])
	);

	logic signed [SUM_ACT_WIDTH-1:0]  psum_shifter_in;
	logic signed [SUM_ACT_WIDTH+6:0]  psum_shifter_out;
	assign psum_shifter_in = psum_mux_out[0] + psum_mux_out[1];
	shifter_3bit #(.IN_WIDTH(SUM_ACT_WIDTH), .OUT_WIDTH(SUM_ACT_WIDTH+7)) shift_psum (
		.in(psum_shifter_in), .shift_sel(column_idx), .out(psum_shifter_out)
	);

	logic signed [DATA_WIDTH-1:0]     hamming_act;
	mux_17to1 #(DATA_WIDTH) mux_hamming (.vec(act), .sel(hamming_sel), .out(hamming_act));

	logic signed [DATA_WIDTH-1:0]  hamming_actin;
	logic signed [DATA_WIDTH+5:0]  hamming_actin_shifted;
	always_comb begin
		if ( hamming_sign == 1'b1 ) begin
			hamming_actin = ~hamming_act + 1;
		end else begin
			hamming_actin = hamming_act;
		end
	end
	shifter_3bit #(.IN_WIDTH(DATA_WIDTH), .OUT_WIDTH(DATA_WIDTH+6)) shift_hamming (
		.in(hamming_actin), .shift_sel(column_idx), .out(hamming_actin_shifted)
	);

	logic signed [SUM_ACT_WIDTH+2:0]  mul_result;
	logic signed [SUM_ACT_WIDTH+5:0]  mul_result_shifted;
	assign mul_result = (sum_act[0] + sum_act[1]) * mul_const;
	shifter_constant #(.IN_WIDTH(SUM_ACT_WIDTH+3), .OUT_WIDTH(SUM_ACT_WIDTH+6)) shift_mul (
		.in(mul_result), .is_shift(is_shift_mul), .out(mul_result_shifted)
	);

	logic signed [SUM_ACT_WIDTH+6:0] psum_special_pe;
	assign psum_special_pe = mul_result_shifted + hamming_actin_shifted;

	logic signed [SUM_ACT_WIDTH+6:0] psum_shifted_tmp, psum_special_pe_tmp, psum_total;
	assign psum_total = psum_shifted_tmp + psum_special_pe_tmp;

	always @(posedge clk) begin
		if (reset) begin
			result <= 0;
		end else if (en) begin
			psum_special_pe_tmp <= psum_special_pe;
			psum_shifted_tmp    <= psum_shifter_out;
			result              <= psum_total + result;
		end
	end

endmodule

`endif