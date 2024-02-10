`ifndef __mux_17to1_V__
`define __mux_17to1_V__

module mux_17to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [15:0], 
    input  logic [4:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);
    always_comb begin
        case (sel) // synopsys infer_mux
            5'b00000: out = vec[0];
            5'b00001: out = vec[1];
            5'b00010: out = vec[2];
            5'b00011: out = vec[3];
            5'b00100: out = vec[4];
            5'b00101: out = vec[5];
            5'b00110: out = vec[6];
            5'b00111: out = vec[7];
            5'b01000: out = vec[8];
            5'b01001: out = vec[9];
            5'b01010: out = vec[10];
            5'b01011: out = vec[11];
            5'b01100: out = vec[12];
            5'b01101: out = vec[13];
            5'b01110: out = vec[14];
            5'b01111: out = vec[15];
            5'b10000: out = 0;
            default:  out = {DATA_WIDTH{1'bx}};
        endcase
    end
endmodule

`endif