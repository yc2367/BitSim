`ifndef __mux_2to1_V__
`define __mux_2to1_V__

module mux_2to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  in_0, 
    input  logic [DATA_WIDTH-1:0]  in_1, 
    input  logic                   sel,     
    output logic [DATA_WIDTH-1:0]  out
);

    always_comb begin
        case (sel) // synopsys infer_mux
            1'b0:    out = in_0;
            1'b1:    out = in_1;
            default: out = {DATA_WIDTH{1'bx}};
        endcase
    end

endmodule

`endif