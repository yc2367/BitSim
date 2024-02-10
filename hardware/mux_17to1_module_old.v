`ifndef __mux_17to1_module_V__
`define __mux_17to1_module_V__

`include "mux_4to1.v"
`include "mux_2to1.v"
`include "mux_5to1_with_zero.v"
`include "mux_3to1_with_zero.v"

module mux_17to1_module
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [15:0], 
    input  logic [4:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);
    genvar i;
    logic [DATA_WIDTH-1:0] middle_value_1 [3:0];
    logic [DATA_WIDTH-1:0] middle_value_2 [1:0];

    generate
	for (i=0; i<4; i=i+1) begin
		mux_4to1 #(DATA_WIDTH) mux1 (
            .vec(vec[4*(i+1)-1:4*i]), .sel(sel[1:0]), .out(middle_value_1[i])
        );
	end
    endgenerate

    generate
	for (i=0; i<2; i=i+1) begin
		mux_2to1 #(DATA_WIDTH) mux2 (
            .vec(middle_value_1[2*(i+1)-1:2*i]), .sel(sel[2]), .out(middle_value_2[i])
        );
	end
    endgenerate

    mux_3to1_with_zero #(DATA_WIDTH) mux3 (.vec(middle_value_2), .sel(sel[4:3]), .out(out));

endmodule

`endif