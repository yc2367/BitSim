`ifndef __pencoder_7to3_V__
`define __pencoder_7to3_V__

module pencoder_7to3
(
    input  logic [6:0]  bitmask,     
    output logic [2:0]  out,
    output logic        val
);
    assign val = (bitmask == 7'd0) ? 1'b0 : 1'b1;

    always_comb begin
        casez (bitmask)
            7'b1000000:  out = 3'b110;
            7'b?100000:  out = 3'b101;
            7'b??10000:  out = 3'b100;
            7'b???1000:  out = 3'b011;
            7'b????100:  out = 3'b010;
            7'b?????10:  out = 3'b001;
            7'b??????1:  out = 3'b000;

            default: out = 3'bxxx;
        endcase
    end

endmodule

`endif