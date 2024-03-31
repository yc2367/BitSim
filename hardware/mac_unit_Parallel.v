`ifndef __mac_unit_Parallel_V__
`define __mac_unit_Parallel_V__


module mac_unit_Parallel
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
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

	logic signed [RESULT_WIDTH-1:0]  mul_out;
	assign mul_out = act_in * w_in;

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
		end else if	(en) begin
			accum_out <= mul_out + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];
endmodule


module mac_unit_Parallel_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
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
		end else if (en) begin
			act_in <= act;
			w_in   <= w;
		end
	end

	mac_unit_Parallel #(DATA_WIDTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif