`ifndef __mac_unit_8_Vert_V__
`define __mac_unit_8_Vert_V__

`include "mux_9to1.v"
`include "mux_4to1.v"


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


module mac_unit_8_Vert
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH) + 1,
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,

	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic        [MUX_SEL_WIDTH-1:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal

	// signal to select an activation that can be calculated wuth hamming distance 
	input  logic        [MUX_SEL_WIDTH-1:0]  hamming_sel,  
	input  logic                             hamming_sign,  

	input  logic signed [SUM_ACT_WIDTH-1:0]  sum_act,       // sum of a group of activations (signed)
	input  logic        [2:0]                column_idx,    // current column index for shifting 
	input  logic        [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic                             is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                             is_msb,        // specify if the current column is MSB
	input  logic                             is_skip_zero,  // specify if skip bit 0
	
	output logic signed [DATA_WIDTH+16:0]    result
);
	genvar j;

	logic [DATA_WIDTH-1:0]     actin     [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	logic [DATA_WIDTH-1:0]     adder_in  [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	generate
	for (j=0; j<VEC_LENGTH/2; j=j+1) begin
		mux_9to1 #(DATA_WIDTH) mux_act (
			.vec(act), .sel(act_sel[j]), .out(actin[j])
		);

		always @(posedge clk) begin
			if (reset) begin
				adder_in[j] <= 0;
			end else begin
				adder_in[j] <= actin[j];
			end
		end
	end
	endgenerate

	logic signed [DATA_WIDTH:0]    psum_1 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+1:0]  psum_actin;
	logic signed [DATA_WIDTH+1:0]  psum_actin_complement;
	generate
		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_1[j] = adder_in[2*j] + adder_in[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_actin = psum_1[2*j] + psum_1[2*j+1];
		end
	endgenerate
	assign psum_actin_complement = ~psum_actin + 1'b1;

	logic signed [SUM_ACT_WIDTH-1:0]  diff_act;
	assign diff_act = sum_act - psum_actin;

	logic signed [SUM_ACT_WIDTH-1:0]  diff_act_complement;
	assign diff_act_complement = psum_actin - sum_act;

	logic [SUM_ACT_WIDTH-1:0] psum_mux_in [3:0];
	assign psum_mux_in[3] = psum_actin_complement; // {is_msb, is_skip_zero}  = 2'b11
	assign psum_mux_in[2] = diff_act_complement;   // {is_msb, is_skip_zero}  = 2'b10
	assign psum_mux_in[1] = psum_actin;            // {is_msb, is_skip_zero}  = 2'b01
	assign psum_mux_in[0] = diff_act;              // {is_msb, is_skip_zero}  = 2'b00

	logic [SUM_ACT_WIDTH-1:0] psum_mux_out;
	logic signed [SUM_ACT_WIDTH+6:0]  psum_shifted;
	mux_4to1 #(SUM_ACT_WIDTH) mux_psum (.vec(psum_mux_in), .sel({is_msb, is_skip_zero}), .out(psum_mux_out));
	shifter_3bit #(.IN_WIDTH(SUM_ACT_WIDTH), .OUT_WIDTH(SUM_ACT_WIDTH+7)) shift_psum (
		.in(psum_mux_out), .shift_sel(column_idx), .out(psum_shifted)
	);

	logic signed [DATA_WIDTH-1:0]     hamming_act;
	mux_9to1 #(DATA_WIDTH) mux_hamming (.vec(act), .sel(hamming_sel), .out(hamming_act));

	logic signed [DATA_WIDTH-1:0]  hamming_actin;
	logic signed [DATA_WIDTH+5:0]  hamming_actin_shifted;
	always_comb begin
		if ( hamming_sign == 1'b1 ) begin
			hamming_actin = ~hamming_act + 1;
		end else begin
			hamming_actin = hamming_act;
		end
	end
	shifter_3bit #(.IN_WIDTH(DATA_WIDTH), .OUT_WIDTH(DATA_WIDTH+6)) shift_hamming_actin (
		.in(hamming_actin), .shift_sel(column_idx), .out(hamming_actin_shifted)
	);

	logic signed [SUM_ACT_WIDTH+2:0]  mul_result;
	logic signed [SUM_ACT_WIDTH+5:0]  mul_result_shifted;
	assign mul_result = sum_act * mul_const;
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
			psum_shifted_tmp    <= psum_shifted;
			result              <= psum_total + result;
		end
	end

endmodule

`endif