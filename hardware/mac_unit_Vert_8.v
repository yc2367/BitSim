`ifndef __mac_unit_Vert_8_V__
`define __mac_unit_Vert_8_V__

`include "mux_5to1.v"


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


module mac_unit_Vert_8
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH),
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                               clk,
	input  logic                               reset,
	input  logic                               en,
	input  logic                               load_accum,

	input  logic signed   [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic          [MUX_SEL_WIDTH-1:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal
	input  logic                               act_val  [VEC_LENGTH/2-1:0], // whether activation is valid
	input  logic signed   [SUM_ACT_WIDTH-1:0]  sum_act, // sum of a group of activations (signed)

	input  logic unsigned [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic          [2:0]                column_idx,    // current column index for shifting 
	input  logic                               is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                               is_msb,        // specify if the current column is MSB
	input  logic                               is_skip_zero,  // specify if skip bit 0
	
	input  logic signed   [ACC_WIDTH-1:0]      accum_prev,
	output logic signed   [RESULT_WIDTH-1:0]   result
);
	genvar i, j;

	logic signed [DATA_WIDTH-1:0] adder_in  [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	generate
		for (j=0; j<4; j=j+1) begin
			mux_5to1 #(DATA_WIDTH) mux_act (
				.vec(act_in[j+4:j]), .sel(act_sel[j]), .val(act_val[j]), .out(adder_in[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH:0]      psum_1        [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+1:0]    psum_act ;
	generate
		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_1[j] = adder_in[2*j] + adder_in[2*j+1];
		end
	
		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_act = psum_1[2*j] + psum_1[2*j+1];
		end
	endgenerate

	logic signed [SUM_ACT_WIDTH-1:0] diff_act ;
	logic signed [SUM_ACT_WIDTH-1:0] psum_act_true ;
	assign diff_act = sum_act - psum_act;
	always_comb begin
		if (is_skip_zero) begin
			psum_act_true = psum_act;
		end else begin
			psum_act_true = diff_act;
		end
	end

	logic signed [SUM_ACT_WIDTH-1:0]  psum_act_shift_in;
	logic signed [SUM_ACT_WIDTH+6:0]  psum_act_shift_out;
	pos_neg_select #(SUM_ACT_WIDTH) twos_complement (.in(psum_act_true), .sign(is_msb), .out(psum_act_shift_in));
	shifter_3bit #(.IN_WIDTH(SUM_ACT_WIDTH), .OUT_WIDTH(SUM_ACT_WIDTH+7)) shift_psum (
		.in(psum_act_shift_in), .shift_sel(column_idx), .out(psum_act_shift_out)
	);

	logic signed [SUM_ACT_WIDTH+1:0]  mul_result;
	logic signed [SUM_ACT_WIDTH+4:0]  mul_result_true;
	assign mul_result = sum_act * mul_const;
	shifter_constant #(.IN_WIDTH(SUM_ACT_WIDTH+2), .OUT_WIDTH(SUM_ACT_WIDTH+5)) shift_mul (
		.in(mul_result), .is_shift(is_shift_mul), .out(mul_result_true)
	);

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	always_comb begin
		if (load_accum) begin
			accum_in = accum_prev;
		end else begin
			accum_in = accum_out;
		end
	end
	
	logic signed [SUM_ACT_WIDTH+4:0] mul_result_true_reg;
	logic signed [SUM_ACT_WIDTH+6:0] psum_act_shift_reg;
	logic signed [SUM_ACT_WIDTH+6:0] sum_act_mul;
	assign sum_act_mul = mul_result_true_reg + psum_act_shift_reg;
	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
			mul_result_true_reg <= 0;
			psum_act_shift_reg  <= 0;
		end else if	(en) begin
			mul_result_true_reg <= mul_result_true;
			psum_act_shift_reg  <= psum_act_shift_out;
			accum_out           <= sum_act_mul + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Vert_8_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH),
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                               clk,
	input  logic                               reset,
	input  logic                               en,
	input  logic                               load_accum,

	input  logic signed   [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic          [MUX_SEL_WIDTH-1:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal
	input  logic                               act_val  [VEC_LENGTH/2-1:0], // whether activation is valid
	input  logic signed   [SUM_ACT_WIDTH-1:0]  sum_act, // sum of a group of activations (signed)

	input  logic unsigned [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic          [2:0]                column_idx,    // current column index for shifting 
	input  logic                               is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                               is_msb,        // specify if the current column is MSB
	input  logic                               is_skip_zero,  // specify if skip bit 0
	
	input  logic signed   [ACC_WIDTH-1:0]      accum_prev,
	output logic signed   [RESULT_WIDTH-1:0]   result
);
	genvar i, j;
	
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

	mac_unit_Vert_8 #(DATA_WIDTH, VEC_LENGTH, MUX_SEL_WIDTH, SUM_ACT_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif