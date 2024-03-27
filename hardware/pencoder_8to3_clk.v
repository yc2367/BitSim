`ifndef __p_encoder_8to3_V__
`define __p_encoder_8to3_V__

module p_encoder_8to3_clk
(
    input  logic [7:0]  bitmas,     
    output logic [2:0]  ou,
    output logic        is_zero,
    input  logic        clk,
    input  logic reset
);
logic [7:0] bitmask;
logic [2:0]  out;
    assign is_zero = (bitmask == 8'd0) ? 1'b1 : 1'b0;

    always_comb begin
        casez (bitmask)
            8'b00000000:  out = 3'b000;
            8'b00000001:  out = 3'b111;
            8'b0000001?:  out = 3'b110;
            8'b000001??:  out = 3'b101;
            8'b00001???:  out = 3'b100;
            8'b0001????:  out = 3'b011;
            8'b001?????:  out = 3'b010;
            8'b01??????:  out = 3'b001;
            8'b1???????:  out = 3'b000;

            default: out = 3'bxxx;
        endcase
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