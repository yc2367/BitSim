`ifndef __mux_17to1_module_V__
`define __mux_17to1_module_V__

`include "mux_4to1.v"
`include "mux_5to1_with_zero.v"

module mux_17to1_module
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [15:0], 
    input  logic [4:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);
    genvar i;
    logic [DATA_WIDTH-1:0] middle_value [3:0];

    generate
	for (i=0; i<4; i=i+1) begin
		mux_4to1 #(DATA_WIDTH) mux1 (
            .vec(vec[4*(i+1)-1:4*i]), .sel(sel[1:0]), .out(middle_value[i])
        );
	end
    endgenerate

    mux_5to1_with_zero #(DATA_WIDTH) mux2 (.vec(middle_value), .sel(sel[4:2]), .out(out));

endmodule

`endif