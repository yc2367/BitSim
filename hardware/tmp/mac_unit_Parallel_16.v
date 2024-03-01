`ifndef __mac_unit_Parallel_16_V__
`define __mac_unit_Parallel_16_V__


module mac_unit_Parallel_16
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 3*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in  [1:0], 
	input  logic signed [DATA_WIDTH-1:0]     w_in    [1:0], 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);

	logic signed [2*DATA_WIDTH-1:0]  mul_out [1:0];
	assign mul_out[0] = act_in[0] * w_in[0];
	assign mul_out[1] = act_in[1] * w_in[1];

	logic signed [2*DATA_WIDTH:0]  psum, psum_reg;
	assign psum = mul_out[0] + mul_out[1];

	always @(posedge clk) begin
		if (reset) begin
			result <= 0;
		end else if	(en) begin
			result <= psum;
		end
	end

endmodule


module mac_unit_Parallel_16_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 3*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act  [1:0], 
	input  logic signed [DATA_WIDTH-1:0]     w    [1:0], 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	
	logic signed [DATA_WIDTH-1:0]  act_in[1:0];
	logic signed [DATA_WIDTH-1:0]  w_in[1:0];
	always @(posedge clk) begin
		if (reset) begin
			act_in[0] <= 0;
			act_in[1] <= 0;
			w_in[0]   <= 0;
			w_in[1]   <= 0;
		end else begin
			act_in <= act;
			w_in   <= w;
		end
	end

	mac_unit_Parallel_16 #(DATA_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif