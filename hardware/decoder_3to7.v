`ifndef __decoder_3to7_V__
`define __decoder_3to7_V__

module decoder_3to7
(
    input  logic [2:0]  in,     
    input  logic        val,     
    output logic [6:0]  out
);
    logic [6:0]  out_tmp;
    always_comb begin
        casez (in)
            3'b000: out_tmp = 7'b0000001;
            3'b001: out_tmp = 7'b0000010;
            3'b010: out_tmp = 7'b0000100;
            3'b011: out_tmp = 7'b0001000;
            3'b100: out_tmp = 7'b0010000;
            3'b101: out_tmp = 7'b0100000;
            3'b110: out_tmp = 7'b1000000;

            default: out_tmp = 7'bxxxxxxx;
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