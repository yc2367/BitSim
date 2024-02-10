`ifndef __mux_4to1_V__
`define __mux_4to1_V__

module mux_4to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [3:0], 
    input  logic [1:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);

    always_comb begin
        case (sel) // synopsys infer_mux
            2'b00:   out = vec[0];
            2'b01:   out = vec[1];
            2'b10:   out = vec[2];
            2'b11:   out = vec[3];
            default: out = {DATA_WIDTH{1'bx}};
        endcase
    end

endmodule

`endif