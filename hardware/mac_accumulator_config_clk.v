`ifndef __mac_accumulator_config_clk_V__
`define __mac_accumulator_config_clk_V__

`include "mac_accumulator_config.v"

module mac_accumulator_config_clk
#(
    parameter DATA_WIDTH = 25,
	parameter VEC_LENGTH = 4
) (
	input  logic                          clk,
	input  logic                          reset,
	input  logic                          en,
	input  logic signed [DATA_WIDTH-1:0]  accu   [VEC_LENGTH-1:0],
	input  logic        [2:0]             se,
	output logic signed [DATA_WIDTH+1:0]  resul
);

	logic signed [DATA_WIDTH-1:0]  accum [VEC_LENGTH-1:0];
    logic [2:0]             sel;
    logic signed [DATA_WIDTH+1:0]  result;

    mac_accumulator_config #(DATA_WIDTH, VEC_LENGTH) m (.*);

	always @(posedge clk) begin
		if (reset) begin
			resul <= 0;
			accum[0] <= 0;
			accum[1] <= 0;
			accum[2] <= 0;
			accum[3] <= 0;
			sel <= 0;
		end else begin
			accum <= accu;
			sel <= se;
			resul <= result;
		end
	end
endmodule

`endif