`ifndef __mac_unit_Vert_16_no_optimization_V__
`define __mac_unit_Vert_16_no_optimization_V__

`include "mux_9to1.v"
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
	parameter OUT_WIDTH = IN_WIDTH + 3
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


module mac_unit_Vert_16_no_optimization
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH) + 1,
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH - 1,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                               clk,
	input  logic                               reset,
	input  logic                               en,
	input  logic                               load_accum,

	input  logic signed   [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic          [MUX_SEL_WIDTH-2:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal
	input  logic signed   [SUM_ACT_WIDTH-1:0]  sum_act  [VEC_LENGTH/8-1:0], // sum of a group of activations (signed)

	input  logic unsigned [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic          [2:0]                column_idx,    // current column index for shifting 
	input  logic                               is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                               is_pooling,    
	input  logic                               is_msb,        // specify if the current column is MSB
	input  logic                               is_skip_zero [VEC_LENGTH/8-1:0],  // specify if skip bit 0
	
	input  logic signed   [RESULT_WIDTH-1:0]   result_prev,
	output logic signed   [RESULT_WIDTH-1:0]   result
);
	genvar i, j;
	localparam PSUM_ACT_WIDTH = SUM_ACT_WIDTH + 1;

	logic signed [DATA_WIDTH-1:0] adder_in  [VEC_LENGTH/2-1:0]; // there are 50% activation to be selected
	generate
		for (i=0; i<VEC_LENGTH/8; i=i+1) begin
			for (j=0; j<VEC_LENGTH/4; j=j+1) begin
				mux_9to1 #(DATA_WIDTH) mux_act (
					.vec(act_in[8*i+7:8*i]), .sel(act_sel[4*i+j]), .out(adder_in[4*i+j])
				);
			end
		end
	endgenerate

	logic signed [DATA_WIDTH:0]      psum_1        [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+1:0]    psum_act      [VEC_LENGTH/8-1:0];
	logic signed [SUM_ACT_WIDTH-1:0] diff_act      [VEC_LENGTH/8-1:0];
	logic signed [SUM_ACT_WIDTH-1:0] psum_act_true [VEC_LENGTH/8-1:0];
	generate
		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_1[j] = adder_in[2*j] + adder_in[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_act[j] = psum_1[2*j] + psum_1[2*j+1];
		end
	
		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign diff_act[j] = sum_act[j] - psum_act[j];

			always_comb begin
				if (is_skip_zero[j]) begin
					psum_act_true[j] = psum_act[j];
				end else begin
					psum_act_true[j] = diff_act[j];
				end
			end
		end
	endgenerate

	logic signed [PSUM_ACT_WIDTH-1:0]  psum_act_total;
	assign psum_act_total = psum_act_true[0] + psum_act_true[1];

	logic signed [PSUM_ACT_WIDTH-1:0]  psum_act_shift_in;
	pos_neg_select #(PSUM_ACT_WIDTH) twos_complement (.in(psum_act_total), .sign(is_msb), .out(psum_act_shift_in));

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	localparam PAD_WIDTH = ACC_WIDTH - RESULT_WIDTH;
	always_comb begin
		if (is_msb) begin
			accum_in = result_prev;
		end else begin
			accum_in = accum_out <<< 1;
		end
	end
	
	logic signed [PSUM_ACT_WIDTH+6:0] psum_act_shift_reg;
	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
		end else if	(en) begin
			psum_act_shift_reg  <= psum_act_shift_in;
			accum_out           <= psum_act_shift_reg + accum_in;
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


module mac_unit_Vert_16_no_optimization_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH) + 1,
	parameter SUM_ACT_WIDTH = $clog2(VEC_LENGTH) + DATA_WIDTH - 1,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                               clk,
	input  logic                               reset,
	input  logic                               en,
	input  logic                               load_accum,

	input  logic signed   [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0],   // input activation (signed)
	input  logic          [MUX_SEL_WIDTH-2:0]  act_sel  [VEC_LENGTH/2-1:0], // input activation MUX select signal
	input  logic signed   [SUM_ACT_WIDTH-1:0]  sum_act  [VEC_LENGTH/8-1:0], // sum of a group of activations (signed)

	input  logic unsigned [2:0]                mul_const,     // constant sent to the multiplier to multiply sum_act

	input  logic          [2:0]                column_idx,    // current column index for shifting 
	input  logic                               is_shift_mul,  // specify whether shift the 3-bit constant multiplier
	input  logic                               is_pooling,    
	input  logic                               is_msb,        // specify if the current column is MSB
	input  logic                               is_skip_zero [VEC_LENGTH/8-1:0],  // specify if skip bit 0
	
	input  logic signed   [RESULT_WIDTH-1:0]   result_prev,
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

	mac_unit_Vert_16_no_optimization #(DATA_WIDTH, VEC_LENGTH, MUX_SEL_WIDTH, SUM_ACT_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif 