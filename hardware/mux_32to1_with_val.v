`ifndef __mux_32to1_with_val_V__
`define __mux_32to1_with_val_V__

module mux_32to1_with_val
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [31:0], 
    input  logic [4:0]             sel,     
    input  logic                   val,     

    output logic [DATA_WIDTH-1:0]  out
);
    logic [DATA_WIDTH-1:0]  out_tmp;
    always_comb begin
        case (sel) // synopsys infer_mux
            5'b00000: out_tmp = vec[0];
            5'b00001: out_tmp = vec[1];
            5'b00010: out_tmp = vec[2];
            5'b00011: out_tmp = vec[3];
            5'b00100: out_tmp = vec[4];
            5'b00101: out_tmp = vec[5];
            5'b00110: out_tmp = vec[6];
            5'b00111: out_tmp = vec[7];
            5'b01000: out_tmp = vec[8];
            5'b01001: out_tmp = vec[9];
            5'b01010: out_tmp = vec[10];
            5'b01011: out_tmp = vec[11];
            5'b01100: out_tmp = vec[12];
            5'b01101: out_tmp = vec[13];
            5'b01110: out_tmp = vec[14];
            5'b01111: out_tmp = vec[15];
            5'b10000: out_tmp = vec[16];
            5'b10001: out_tmp = vec[17];
            5'b10010: out_tmp = vec[18];
            5'b10011: out_tmp = vec[19];
            5'b10100: out_tmp = vec[20];
            5'b10101: out_tmp = vec[21];
            5'b10110: out_tmp = vec[22];
            5'b10111: out_tmp = vec[23];
            5'b11000: out_tmp = vec[24];
            5'b11001: out_tmp = vec[25];
            5'b11010: out_tmp = vec[26];
            5'b11011: out_tmp = vec[27];
            5'b11100: out_tmp = vec[28];
            5'b11101: out_tmp = vec[29];
            5'b11110: out_tmp = vec[30];
            5'b11111: out_tmp = vec[31];
            default:  out_tmp = {DATA_WIDTH{1'bx}};
        endcase
    end

    always_comb begin
        if ( val ) begin
            out = out_tmp;
        end else begin
            out = 0;
        end
    end
endmodule

`endif