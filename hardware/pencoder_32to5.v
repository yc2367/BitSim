`ifndef __pencoder_32to5_V__
`define __pencoder_32to5_V__

module pencoder_32to5
(
    input  logic [31:0] bitmask,     
    output logic [4:0]  out,
    output logic        val
);
    assign val = (bitmask == 32'd0) ? 1'b0 : 1'b1;

    always_comb begin
        casez (bitmask)
            32'b00000000000000000000000000000001:  out = 5'b11111;
            32'b0000000000000000000000000000001?:  out = 5'b11110;
            32'b000000000000000000000000000001??:  out = 5'b11101;
            32'b00000000000000000000000000001???:  out = 5'b11100;
            32'b0000000000000000000000000001????:  out = 5'b11011;
            32'b000000000000000000000000001?????:  out = 5'b11010;
            32'b00000000000000000000000001??????:  out = 5'b11001;
            32'b0000000000000000000000001???????:  out = 5'b11000;
            32'b000000000000000000000001????????:  out = 5'b10111;
            32'b00000000000000000000001?????????:  out = 5'b10110;
            32'b0000000000000000000001??????????:  out = 5'b10101;
            32'b000000000000000000001???????????:  out = 5'b10100;
            32'b00000000000000000001????????????:  out = 5'b10011;
            32'b0000000000000000001?????????????:  out = 5'b10010;
            32'b000000000000000001??????????????:  out = 5'b10001;
            32'b00000000000000001???????????????:  out = 5'b10000;

            32'b0000000000000001????????????????:  out = 5'b01111;
            32'b000000000000001?????????????????:  out = 5'b01110;
            32'b00000000000001??????????????????:  out = 5'b01101;
            32'b0000000000001???????????????????:  out = 5'b01100;
            32'b000000000001????????????????????:  out = 5'b01011;
            32'b00000000001?????????????????????:  out = 5'b01010;
            32'b0000000001??????????????????????:  out = 5'b01001;
            32'b000000001???????????????????????:  out = 5'b01000;
            32'b00000001????????????????????????:  out = 5'b00111;
            32'b0000001?????????????????????????:  out = 5'b00110;
            32'b000001??????????????????????????:  out = 5'b00101;
            32'b00001???????????????????????????:  out = 5'b00100;
            32'b0001????????????????????????????:  out = 5'b00011;
            32'b001?????????????????????????????:  out = 5'b00010;
            32'b01??????????????????????????????:  out = 5'b00001;
            32'b1???????????????????????????????:  out = 5'b00000;

            default: out = 5'bxxxxx;
        endcase
    end

endmodule

`endif