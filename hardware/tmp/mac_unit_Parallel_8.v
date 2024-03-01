`ifndef __mac_unit_Parallel_8_V__
`define __mac_unit_Parallel_8_V__


module mac_unit_Parallel_8
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 3*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in, 
	input  logic signed [DATA_WIDTH-1:0]     w_in, 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);

	logic signed [2*DATA_WIDTH-1:0]  mul_out;
	assign mul_out = act_in * w_in;


	always @(posedge clk) begin
		if (reset) begin
			result <= 0;
		end else if	(en) begin
			result <= mul_out;
		end
	end

endmodule


module mac_unit_Parallel_8_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 3*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act, 
	input  logic signed [DATA_WIDTH-1:0]     w, 
	input  logic signed [ACC_WIDTH-1:0]      accum_prev,

	output logic signed [RESULT_WIDTH-1:0]   result
);
	
	logic signed [DATA_WIDTH-1:0]  act_in;
	logic signed [DATA_WIDTH-1:0]  w_in;
	always @(posedge clk) begin
		if (reset) begin
			act_in <= 0;
			w_in   <= 0;
		end else begin
			act_in <= act;
			w_in   <= w;
		end
	end

	mac_unit_Parallel_8 #(DATA_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif