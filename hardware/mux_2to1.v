`ifndef __mux_2to1_V__
`define __mux_2to1_V__

module mux_2to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [1:0], 
    input  logic                   sel,     
    output logic [DATA_WIDTH-1:0]  out
);

    always_comb begin
        case (sel)
            1'b0:    out = vec[0];
            1'b1:    out = vec[1];
            default: out = {DATA_WIDTH{1'bx}};
        endcase
    end

endmodule

`endif