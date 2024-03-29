`ifndef __mac_unit_Bitlet_16_V__
`define __mac_unit_Bitlet_16_V__

`include "mux_16to1_with_val.v"

module twos_complement #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	output logic signed [DATA_WIDTH:0]   out
); 
	assign out = ~in + 1'b1;
endmodule


module mac_unit_Bitlet_16
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH),
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	// act_in:  activation
	// act_sel: activation MUX select signal
	// act_val: whether activation is valid
	input  logic signed [DATA_WIDTH-1:0]     act_in   [VEC_LENGTH-1:0], 
	input  logic        [MUX_SEL_WIDTH-1:0]  act_sel  [DATA_WIDTH-1:0], 
	input  logic                             act_val  [DATA_WIDTH-1:0], 

	input  logic signed [ACC_WIDTH-1:0]      accum_prev, // previous accumulator partial sum

	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_out  [DATA_WIDTH-1:0] ;
	generate
		for (j=0; j<DATA_WIDTH; j=j+1) begin
			mux_16to1_with_val #(DATA_WIDTH) mux_act (
				.vec(act_in), .sel(act_sel[j]), .val(act_val[j]), .out(act_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH-1:0]  adder_in_0;
	logic signed [DATA_WIDTH+0:0]  adder_in_1;
	logic signed [DATA_WIDTH+1:0]  adder_in_2;
	logic signed [DATA_WIDTH+2:0]  adder_in_3;
	logic signed [DATA_WIDTH+3:0]  adder_in_4;
	logic signed [DATA_WIDTH+4:0]  adder_in_5;
	logic signed [DATA_WIDTH+5:0]  adder_in_6;
	logic signed [DATA_WIDTH+6:0]  adder_in_7_tmp;
	logic signed [DATA_WIDTH+7:0]  adder_in_7;
	assign adder_in_0 = act_out[0];
	assign adder_in_1 = act_out[1] <<< 1;
	assign adder_in_2 = act_out[2] <<< 2;
	assign adder_in_3 = act_out[3] <<< 3;
	assign adder_in_4 = act_out[4] <<< 4;
	assign adder_in_5 = act_out[5] <<< 5;
	assign adder_in_6 = act_out[6] <<< 6;
	assign adder_in_7_tmp = act_out[7] <<< 7;

	twos_complement #(DATA_WIDTH+7) complement (.in(adder_in_7_tmp), .out(adder_in_7));

	logic signed [DATA_WIDTH+1:0]  psum_01;
	logic signed [DATA_WIDTH+3:0]  psum_23;
	logic signed [DATA_WIDTH+5:0]  psum_45;
	logic signed [DATA_WIDTH+7:0]  psum_67;
	assign psum_01 = adder_in_0 + adder_in_1;
	assign psum_23 = adder_in_2 + adder_in_3;
	assign psum_45 = adder_in_4 + adder_in_5;
	assign psum_67 = adder_in_6 + adder_in_7;

	logic signed [DATA_WIDTH+4:0]  psum_1234;
	logic signed [DATA_WIDTH+7:0]  psum_4567;
	assign psum_1234 = psum_01 + psum_23;
	assign psum_4567 = psum_45 + psum_67;

	logic signed [DATA_WIDTH+7:0]  psum_total, psum_total_reg;
	assign psum_total = psum_1234 + psum_4567;

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
		end else if	(en) begin
			psum_total_reg <= psum_total;
			accum_out      <= psum_total_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];
endmodule


module mac_unit_Bitlet_16_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 16,
	parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH),
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	// act_in:  activation
	// act_sel: activation MUX select signal
	// act_val: whether activation is valid
	input  logic signed [DATA_WIDTH-1:0]     act      [VEC_LENGTH-1:0], 
	input  logic        [MUX_SEL_WIDTH-1:0]  act_sel  [DATA_WIDTH-1:0], 
	input  logic                             act_val  [DATA_WIDTH-1:0], 

	input  logic signed [ACC_WIDTH-1:0]      accum_prev, // previous accumulator partial sum

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

	mac_unit_Bitlet_16 #(DATA_WIDTH, VEC_LENGTH, MUX_SEL_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif