`ifndef __p_encoder_16to4_module_V__
`define __p_encoder_16to4_module_V__

`include "p_encoder_8to3.v"

module p_encoder_16to4_module
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
    logic        valid_H, valid_L;
    assign bitmask_H = bitmask[15:8];
    assign bitmask_L = bitmask[7:0];
    p_encoder_8to3 encoder_H (.bitmask(bitmask_H), .out(out_H), .is_zero(valid_H));
    p_encoder_8to3 encoder_L (.bitmask(bitmask_L), .out(out_L), .is_zero(valid_L));
    assign is_zero = valid_H | valid_L;

    always_comb begin
        if ( valid_H ) begin
            out = {1'b0, out_H};
        end else if ( valid_L ) begin 
            out = {1'b1, out_L};
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