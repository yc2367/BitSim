`ifndef __mac_unit_adder_tree_8_Vert_V__
`define __mac_unit_adder_tree_8_Vert_V__

`include "mux_4to1.v"

module mac_unit_adder_tree_8_Vert
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 4,
	parameter SUM_ACT_WIDTH = 11
) (
	input  logic signed [DATA_WIDTH-1:0]     adder_in [VEC_LENGTH-1:0],  // input activation (signed)
	input  logic signed [SUM_ACT_WIDTH-1:0]  sum_act,                    // sum of a group of activations (signed)

	input  logic                             is_msb,        // specify if the current column is MSB
	input  logic                             is_skip_zero,  // specify if skip bit 0
	
	output logic signed [SUM_ACT_WIDTH-1:0]  result
);
	genvar j;

	logic signed [DATA_WIDTH:0]      psum_1          [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+1:0]    psum_adder_tree;
	logic signed [DATA_WIDTH+1:0]    psum_adder_tree_complement;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = adder_in[2*j] + adder_in[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_adder_tree = psum_1[2*j] + psum_1[2*j+1];
		end
	endgenerate
	assign psum_adder_tree_complement = ~psum_adder_tree + 1'b1;

	logic signed [SUM_ACT_WIDTH-1:0] diff_act;
	logic signed [SUM_ACT_WIDTH-1:0] diff_act_complement;
	assign diff_act = sum_act - psum_adder_tree;
	assign diff_act_complement = psum_adder_tree - sum_act;

	logic signed [SUM_ACT_WIDTH-1:0] psum_mux_in [3:0];
	assign psum_mux_in[3] = psum_adder_tree_complement; // {is_msb, is_skip_zero}  = 2'b11
	assign psum_mux_in[2] = diff_act_complement;        // {is_msb, is_skip_zero}  = 2'b10
	assign psum_mux_in[1] = psum_adder_tree;            // {is_msb, is_skip_zero}  = 2'b01
	assign psum_mux_in[0] = diff_act;                   // {is_msb, is_skip_zero}  = 2'b00

	logic [SUM_ACT_WIDTH-1:0] psum_mux_out;
	mux_4to1 #(SUM_ACT_WIDTH) mux_psum (.vec(psum_mux_in), .sel({is_msb, is_skip_zero}), .out(psum_mux_out));
	assign result = psum_mux_out;

endmodule

`endif