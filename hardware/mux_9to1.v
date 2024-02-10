`ifndef __mux_9to1_V__
`define __mux_9to1_V__

module mux_9to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [7:0], 
    input  logic [3:0]             sel,    
    output logic [DATA_WIDTH-1:0]  out
);
    always_comb begin
        case (sel) // synopsys infer_mux
            4'b0000: out = vec[0];
            4'b0001: out = vec[1];
            4'b0010: out = vec[2];
            4'b0011: out = vec[3];
            4'b0100: out = vec[4];
            4'b0101: out = vec[5];
            4'b0110: out = vec[6];
            4'b0111: out = vec[7];
            4'b1000: out = 0;
            default:  out = {DATA_WIDTH{1'bx}};
        endcase
    end
    
endmodule

`endif