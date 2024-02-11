`ifndef __mux_17to1_clk_V__
`define __mux_17to1_clk_V__

`include "mux_17to1.v"

module mux_17to1_clk
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  ve [15:0], 
    input  logic [4:0]             se,     
    output logic [DATA_WIDTH-1:0]  ou,
    input  logic        clk,
    input  logic reset
);
    logic [DATA_WIDTH-1:0]  vec [15:0];
    logic [4:0]             sel;
    logic [DATA_WIDTH-1:0]  out;

    mux_17to1 #(DATA_WIDTH) m (.*);

always @(posedge clk) begin
		if (reset) begin
			ou <= 0;
		end else begin
			ou <= out;
            sel <= se;
		end
	end
endmodule

`endif