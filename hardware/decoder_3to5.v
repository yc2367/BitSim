`ifndef __decoder_3to5_V__
`define __decoder_3to5_V__

module decoder_3to5
(
    input  logic [2:0]  in,     
    input  logic        val,     
    output logic [4:0]  out
);
    logic [4:0]  out_tmp;
    always_comb begin
        casez (in)
            3'b000: out_tmp = 5'b10000;
            3'b001: out_tmp = 5'b01000;
            3'b010: out_tmp = 5'b00100;
            3'b011: out_tmp = 5'b00010;
            3'b100: out_tmp = 5'b00001;

            default: out_tmp = 5'bxxxxx;
        endcase
    end

    always_comb begin
        if (val) begin
            out = out_tmp;
        end else begin
            out = 0;
        end
    end
endmodule

`endif