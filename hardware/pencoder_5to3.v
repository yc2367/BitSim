`ifndef __pencoder_5to3_V__
`define __pencoder_5to3_V__

module pencoder_5to3
(
    input  logic [4:0]  bitmask,     
    output logic [2:0]  out,
    output logic        val
);
    assign val = (bitmask == 5'd0) ? 1'b0 : 1'b1;

    always_comb begin
        casez (bitmask)
            5'b00001:  out = 3'b100;
            5'b0001?:  out = 3'b011;
            5'b001??:  out = 3'b010;
            5'b01???:  out = 3'b001;
            5'b1????:  out = 3'b000;

            default:   out = 3'bxxx;
        endcase
    end

endmodule

`endif