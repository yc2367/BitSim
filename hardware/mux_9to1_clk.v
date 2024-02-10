`ifndef __mux_9to1_clk_V__
`define __mux_9to1_clk_V__

`include "mux_9to1.v"

module mux_9to1_clk
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  ve [7:0], 
    input  logic [3:0]             se,     
    output logic [DATA_WIDTH-1:0]  ou,
    input  logic        clk,
    input  logic reset
);
    logic [DATA_WIDTH-1:0]  vec [7:0];
    logic [3:0]             sel;
    logic [DATA_WIDTH-1:0]  out;

    mux_9to1 #(DATA_WIDTH) m (.*);

always @(posedge clk) begin
		if (reset) begin
			ou <= 0;
		end else begin
			ou <= out;
            sel <= se;
            vec <= ve;
		end
	end
endmodule

`endif