`ifndef __p_encoder_16to4_module_clk_V__
`define __p_encoder_16to4_module_clk_V__

`include "p_encoder_8to3.v"

module p_encoder_16to4_module_clk
(
    input  logic [15:0] bitmas,     
    output logic [3:0]  ou,
    output logic        is_zero,
    input  logic        clk,
    input  logic reset
);
logic [15:0] bitmask;
logic [3:0]  out;

    logic [2:0]  out_H, out_L;
    logic [7:0]  bitmask_H, bitmask_L;
    logic        is_zero_H, is_zero_L;
    assign bitmask_L = bitmask[15:8];
    assign bitmask_H = bitmask[7:0];
    p_encoder_8to3 encoder_H (.bitmas(bitmask_H), .ou(out_H), .is_zero(is_zero_H), .*);
    p_encoder_8to3 encoder_L (.bitmas(bitmask_L), .ou(out_L), .is_zero(is_zero_L), .*);
    assign is_zero = is_zero_H & is_zero_L;

    always_comb begin
        if ( !is_zero_H ) begin
            out = {1'b1, out_H};
        end else if ( !is_zero_L ) begin 
            out = {1'b0, out_L};
        end else begin
            out = 4'b0000;
        end
    end

    always @(posedge clk) begin
		if (reset) begin
			ou <= 0;
		end else begin
			ou <= out;
            bitmask <= bitmas;
		end
	end

endmodule

`endif