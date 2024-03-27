`ifndef __pencoder_16to4_V__
`define __pencoder_16to4_V__

module pencoder_16to4_clk
(
    input  logic [15:0] bitmas,     
    output logic [3:0]  ou,
    output logic        is_zero,
    input  logic        clk,
    input  logic reset
);
logic [15:0] bitmask;
logic [3:0]  out;
    assign is_zero = (bitmask == 16'd0) ? 1'b1 : 1'b0;

    always_comb begin
        casez (bitmask)
            16'b0000000000000000:  out = 4'b0000;
            16'b0000000000000001:  out = 4'b1111;
            16'b000000000000001?:  out = 4'b1110;
            16'b00000000000001??:  out = 4'b1101;
            16'b0000000000001???:  out = 4'b1100;
            16'b000000000001????:  out = 4'b1011;
            16'b00000000001?????:  out = 4'b1010;
            16'b0000000001??????:  out = 4'b1001;
            16'b000000001???????:  out = 4'b1000;
            16'b00000001????????:  out = 4'b0111;
            16'b0000001?????????:  out = 4'b0110;
            16'b000001??????????:  out = 4'b0101;
            16'b00001???????????:  out = 4'b0100;
            16'b0001????????????:  out = 4'b0011;
            16'b001?????????????:  out = 4'b0010;
            16'b01??????????????:  out = 4'b0001;
            16'b1???????????????:  out = 4'b0000;

            default: out = 4'bxxxx;
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