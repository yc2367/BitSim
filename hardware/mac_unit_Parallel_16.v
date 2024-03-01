`ifndef __mac_unit_Parallel_16_V__
`define __mac_unit_Parallel_16_V__


module mac_unit_Parallel_16
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
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
			psum_reg  <= 0;
		end else if	(en) begin
			psum_reg <= psum;
			accum_out <= psum_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Parallel_16_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
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